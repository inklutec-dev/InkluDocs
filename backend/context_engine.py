"""
Context Engine for InkluDocs Alt-Text Generation.
Version 2.2 – Anti-Halluzination + Funktional + Thumbnail-Guard (30.03.2026)

Pipeline:
  Stufe 1 (Qwen, lokal): Klassifikation (Bildtyp + dekorativ + konfidenz + original_alt_brauchbar)
  Stufe 2 (Mistral, API): Alt-Text-Generierung mit spezialisierten Prompts
  Fallback: Qwen generiert wenn Mistral fehlschlaegt

Modi (PIPELINE_MODE Umgebungsvariable):
  mistral_primary: Qwen klassifiziert, Mistral generiert (Default, beste Qualitaet)
  hybrid: Qwen macht einfache Bilder, Mistral die komplexen (spart Kosten)
  qwen_only: Alles ueber Qwen (Fallback, niedrigste Kosten)

v2.2 Changes:
  - New category "funktional" for navigation/UI elements
  - Thumbnail guard: images < 200px get simplified prompts
  - Improvement mode: good original alt-texts are preserved/improved, not replaced
  - Anti-hallucination: stricter rules for context usage
  - Dekorativ validation: safety net to catch misclassified functional elements
"""

import os
import re

PIPELINE_MODE = os.environ.get("PIPELINE_MODE", "mistral_primary")

# ─── Thumbnail-Schwellwerte ────────────────────────────────────
MIN_DETAIL_WIDTH = 200
MIN_DETAIL_HEIGHT = 200

# ─── Klassifikations-Prompt (Stufe 1, Qwen) v2.2 ─────────────
CLASSIFICATION_PROMPT = """/no_think
Klassifiziere dieses Bild. Antworte NUR mit diesem JSON:
{{"bildtyp": "foto|diagramm|tabelle|strukturformel|logo|icon|funktional|karte|screenshot|infografik|dekorativ", "ist_dekorativ": true|false, "konfidenz": "hoch|mittel|niedrig", "original_alt_brauchbar": true|false}}

Bildgroesse: {width}x{height} Pixel
Originaler Alt-Text: {original_alt}

Regeln:
- dekorativ = rein abstrakte Formen, Farbverlaeufe, Trennlinien, Schmuckelemente ohne jede Information
- Wenn Text, Personen, Daten, Diagramme oder konkrete Objekte sichtbar sind: NICHT dekorativ
- icon = einzelnes kleines Symbol mit funktionaler Bedeutung (Lupe, Hamburger-Menue, Warenkorb)
- funktional = Navigationselemente mit Zustandsinformation: Paginierungspfeile, Vor/Zurueck-Buttons, Fortschrittsanzeigen, Breadcrumbs. Diese sind NICHT dekorativ, auch wenn sie einfache Grafiken sind. Hinweis: Wenn der originale Alt-Text eine Funktion beschreibt ("Keine vorherige Seite", "Naechste Seite", "Zurueck"), ist es fast immer funktional.
- logo = erkennbare Marken-, Organisations- oder Lizenzlogos (auch Creative Commons, Zertifizierungssiegel, Guetesiegel)
- original_alt_brauchbar = true wenn der originale Alt-Text bereits eine sinnvolle, spezifische Beschreibung enthaelt (NICHT nur "Bild", "Foto", "Grafik" oder leer). Ein brauchbarer Alt-Text beschreibt konkret was zu sehen ist oder welche Funktion das Element hat.
- konfidenz: hoch = eindeutig, mittel = wahrscheinlich, niedrig = unklar

Kontext: {context}"""

# ─── Spezialisierte Generierungs-Prompts (Stufe 2, Mistral) ──

SPECIALIZED_GENERATION_PROMPTS = {

    "foto": """Du bist ein Experte fuer barrierefreie Bildbeschreibungen nach WCAG 2.2.
Dieses Bild ist ein FOTO.
Bildgroesse: {width}x{height} Pixel

{thumbnail_block}
{verbesserung_block}

DEIN AUFTRAG: Vermittle nicht was das Bild ZEIGT, sondern was es BEDEUTET. Was wuerde ein Sehender sofort denken und wissen? Genau das muss der blinde Nutzer erfahren.

EVIDENZ-BASIERTE IDENTIFIKATION:
- STUFE 1 (IMMER): Text, Namen, Logos die im Bild KLAR LESBAR sind → direkt nennen.
- STUFE 2 (ERLAUBT): Lesbarer Text oder eindeutiges Logo + Allgemeinwissen → benennen. Beispiel: Inschrift "EQUAL JUSTICE UNDER LAW" → "Supreme Court der USA". Mercedes-Stern sichtbar → "Mercedes-Benz". Aber Fahrzeugmodell NUR wenn als Text lesbar oder aus Kontext eindeutig.
- STUFE 3 (VERBOTEN): Kein Text, kein Logo, nur visueller Eindruck → allgemein beschreiben. "Ein industrielles Steuerungsmodul", NICHT "Siemens". "Eine Person", NICHT einen Namen raten.

ANTI-HALLUZINATION – KONTEXTNUTZUNG:
- Der Seitenkontext (Titel, Ueberschriften, Meta-Beschreibung) hilft dir, das THEMA zu verstehen.
- Der Kontext erlaubt dir NICHT, Details ins Bild hineinzuinterpretieren die du nicht siehst.
- VERBOTEN: Aus dem Kontext "Bundesanstalt fuer Landwirtschaft" zu schliessen, dass ein unscharfes Bild "nachhaltige Landwirtschaft symbolisiert".
- ERLAUBT: Aus dem Kontext zu verstehen, dass ein Feld auf einer Landwirtschaftsseite thematisch zur Seite gehoert.
- FAUSTREGEL: Wenn du einen Bildinhalt nur wegen des Kontexts beschreibst, aber nicht weil du ihn SIEHST – lass ihn weg.

VERBOTENE FORMULIERUNGEN: Verwende NICHT 'symbolisiert', 'steht symbolisch fuer', 'repraesentiert', 'steht fuer', 'thematisch passend zu', 'im Kontext von', 'vermutlich im Zusammenhang mit'. Wenn du zu diesen Formulierungen greifst, beschreibst du den Kontext statt das Bild – formuliere um und beschreibe was du SIEHST.

KEINE VERMUTUNGEN (KRITISCH):
- VERBOTEN in Alt-Texten: vermutlich, wahrscheinlich, moeglicherweise, koennte, duerfte, scheint, wohl, anscheinend, offenbar.
- VERBOTEN: Symbol fuer, im Rahmen von, im Rahmen der/des/eines.
- Alt-Text beschreibt was zu SEHEN ist, nicht was etwas bedeuten KOENNTE.
- Wenn du unsicher bist was etwas ist, beschreibe das AUSSEHEN: Grosses Gebaeude mit gruenen Pfeilern statt vermutlich eine Sporthalle.
- Wenn du den Zweck nicht kennst, lass ihn weg: Feld mit gelben Blueten statt Rapsfeld, das nachhaltige Landwirtschaft symbolisiert.

REGELN:
- KEIN Praefix "Foto – ". Starte direkt mit der Erkenntnis.
- Personen: Name und Funktion NUR aus dem Kontext oder von lesbaren Namensschildern. KEIN Alter nennen.
- Gebaeude/Orte: Benennen wenn Evidenz vorhanden (Schild, Inschrift, Kontext). Sonst allgemein.
- Gruppen: Ungefaehre Anzahl, Anlass, Setting – was PASSIERT hier?
- VERLINKTE BILDER: Wenn [Link-Ziel] im Kontext steht → der Alt-Text MUSS das Link-Ziel als Information enthalten. Format: "Beschreibung (verweist auf: Link-Ziel)". Das ist fuer Screenreader-Nutzer essentiell zur Navigation.
- KONTEXT-IDENTIFIKATION (WICHTIG): Wenn der Kontext den NAMEN oder die FUNKTION einer Person nennt, MUSS dieser Name im Alt-Text stehen! Der Alt-Text muss allein verstaendlich sein – der Nutzer sieht den Kontext nicht.
- ANTI-REDUNDANZ: Wiederhole keine BESCHREIBENDEN Details die der Kontext bereits nennt. Aber Namen, Funktionen und Identitaeten IMMER nennen – sie sind die Kerninfo.
- Unterschriften: Verwende den GEDRUCKTEN Namen neben oder unter der Unterschrift. Versuche NIEMALS die Handschrift selbst zu entziffern oder einen Namen daraus abzulesen. Handschriftliche Unterschriften sind per Definition nicht maschinenlesbar.
- Farben NUR wenn sie Information tragen (Warnschilder, Signalfarben). Keine optischen Farben.
- Sprache: Natuerliches Deutsch. Kein Aufzaehlungsstil. Wie ein Mensch es einem anderen erzaehlen wuerde.

Antworte NUR mit diesem JSON:
{{"alt_text": "...", "langbeschreibung": "..."}}

alt_text: Max 250 Zeichen. Die Kernaussage – was ein Sehender sofort erkennt und denkt.
langbeschreibung: Ergaenzende Details die den Alt-Text vertiefen. Max 800 Zeichen. LEER LASSEN wenn der Alt-Text bereits alles Wesentliche sagt oder wenn das Bild zu klein fuer weitere Details ist.

Kontext: {context}""",

    "diagramm": """Du bist ein Experte fuer barrierefreie Bildbeschreibungen nach WCAG 2.2.
Dieses Bild ist ein DIAGRAMM.

DEIN AUFTRAG – INSIGHT FIRST: Ein Sehender schaut auf ein Diagramm und erkennt sofort den TREND, das MUSTER, die KERNAUSSAGE. Genau DAS muss dein erster Satz sein. Nicht die Achsenbeschriftung. Nicht der Diagrammtyp. Sondern: Was sagt uns dieses Diagramm?

ABER ZUERST – INVENTAR-PFLICHT (KRITISCH):
Bevor du auch nur einen Trend formulierst, erfasse VOLLSTAENDIG:
1. Diagrammtitel (wenn vorhanden, IMMER nennen)
2. ALLE Kategorien auf der X-Achse – zaehle sie durch
3. ALLE Eintraege in der Legende – zaehle sie durch
4. Y-Achse: Einheit und Skalierung
5. Anzahl der Balken/Linien/Segmente pro Kategorie

Wenn du z.B. 4 Kategorien auf der X-Achse siehst (Software, Hardware, Services, Mobile), dann muessen ALLE VIER in alt_text und langbeschreibung vorkommen. Wenn du nur 2 von 4 erwaeehnst, ist die Beschreibung FALSCH und UNVOLLSTAENDIG.

TREND-VALIDIERUNG (KRITISCH):
- Beschreibe Trends NUR nach paarweisem Vergleich ALLER Datenpunkte.
- VERBOTEN: Woerter wie "kontinuierlich", "stetig", "durchgehend" ohne JEDEN Datenpunkt geprueft zu haben.
- RICHTIG: "Von 2021 zu 2022 steigt der Wert, von 2022 zu 2023 faellt er wieder" statt "kontinuierlicher Anstieg".
- Wenn 3 Jahre dargestellt sind, vergleiche: 2021→2022, 2022→2023, UND 2021→2023. Erst dann darfst du einen Gesamttrend formulieren.
- GESAMTTREND = Vergleich ERSTER Wert mit LETZTEM Wert. Wenn 2021=4 und 2023=2, ist der Gesamttrend ABWAERTS, auch wenn 2022 dazwischen anders lag.
- Pruefe bei JEDER Kategorie separat. Nicht von einer Kategorie auf eine andere schliessen.

ANTI-ERFINDUNG (KRITISCH):
- Behaupte NIEMALS, dass Werte "unleserlich", "ueberlappend" oder "nicht erkennbar" sind, es sei denn sie sind es TATSAECHLICH.
- Wenn die Y-Achse klar skaliert ist (z.B. 0-6), dann SAGE das. Erfinde keine Lesbarkeitsprobleme.
- Wenn du einen Wert nicht exakt ablesen kannst, formuliere: "liegt bei circa X" – NICHT "ist unleserlich".

KONSISTENZ-PFLICHT (KRITISCH):
- Analysiere ZUERST alle Datenpunkte pro Kategorie und notiere die Trends.
- Formuliere den alt_text DANACH basierend auf dieser Analyse.
- PRUEFE BEVOR DU ANTWORTEST: Stimmen die Trends im alt_text mit den Zahlen in der langbeschreibung ueberein? Wenn der alt_text "Zuwachs" oder "Anstieg" fuer eine Kategorie behauptet, muessen die Zahlen in der langbeschreibung das bestaetigen. Wenn nicht: alt_text KORRIGIEREN.
- VERBOTEN: Fuer eine Kategorie im alt_text "steigt" oder "Zuwachs" sagen, wenn die Werte von 2021 zu 2023 insgesamt SINKEN (auch wenn es zwischendurch schwankt).
- BEISPIEL FUER FEHLER: alt_text sagt "Software zeigt 2023 Zuwaechse" aber Zahlen zeigen 4.0 → 2.5 → 2.0 = Rueckgang. Das ist ein WIDERSPRUCH und VERBOTEN.

BEISPIELE FUER GUTE INSIGHTS:
- "Balkendiagramm – Umsatzentwicklung in Deutschland: Hardware erreicht 2022 den Hoechstwert und faellt 2023 deutlich, waehrend Mobile 2023 den staerksten Zuwachs aller vier Kategorien zeigt."
- "Kreisdiagramm – Fast die Haelfte der Buerokratie-Entlastung entfaellt auf ein einziges Gesetz, das Wachstumschancengesetz mit 39%."

EVIDENZ-BASIERTE IDENTIFIKATION:
- Nenne NUR Zahlen, Beschriftungen und Legenden die du KLAR LESEN kannst.
- Wenn OCR-Text bereitgestellt wird ([OCR-Text im Bild]), nutze diesen als primaere Datenquelle.

REGELN:
- alt_text: Diagrammtyp + Bindestrich + Titel (wenn vorhanden) + Kernaussage/Trend ueber ALLE Kategorien. 2-3 Saetze, max 350 Zeichen.
- langbeschreibung: ALLE Kategorien und ALLE Legendeneintraege benennen. Dann pro Kategorie die lesbaren Datenpunkte auflisten. Achsenbeschriftungen, Legendenwerte, Anfangs-/Endwerte bei Zeitreihen. Hoechst- und Tiefstwerte benennen. Fliesstext oder strukturierte Liste, KEINE Markdown-Tabellen. Max 1500 Zeichen.
- Sprache: Deutsch. Insight zuerst, Details danach.
- ANTI-HALLUZINATION: Erfinde KEINE Zahlen. Erfinde KEINE visuellen Maengel. Lass keine Kategorien weg.

Antworte NUR mit diesem JSON:
{{"alt_text": "...", "langbeschreibung": "..."}}

Kontext: {context}""",

    "tabelle": """Du bist ein Experte fuer barrierefreie Bildbeschreibungen nach WCAG 2.2.
Dieses Bild zeigt eine TABELLE als Grafik.

DEIN AUFTRAG – INSIGHT FIRST: Was ist die ERKENNTNIS dieser Tabelle? Welchen Schluss zieht ein Sehender auf den ersten Blick? Fuehre damit. Dann die Daten.

ABER ZUERST – SPALTEN-ZUORDNUNG (KRITISCH):
1. Lies ZUERST alle Spaltenkoepfe von links nach rechts und notiere sie.
2. Lies DANN jede Zeile und ordne JEDEN Wert seiner EXAKTEN Spalte zu.
3. Verwechsle NIEMALS Zwischenwerte (z.B. Zugaenge, Abschreibungen, Veraenderungen) mit Bestands- oder Endwerten.
4. Wenn eine Zeile Werte in der Spalte "01.01." UND "31.12." hat, sind das VERSCHIEDENE Werte – nenne BEIDE mit Spaltenzuordnung.
5. Bei Buchhaltungstabellen: Anfangsbestand und Endbestand sind die entscheidenden Werte, NICHT die Bewegungen dazwischen.

BEISPIEL FUER KORREKTE ZUORDNUNG:
- Spalten: "Aktiva | 01.01. | Zugaenge | Abschreibungen | 31.12.2023"
- Zeile: "Ausstattung | 1,00 | 39.891.626,46 | 39.891.626,46 | 1,00"
- RICHTIG: "Betriebs- und Geschaeftsausstattung: Anfangsbestand 1,00 EUR, Zugaenge und Abschreibungen jeweils 39,9 Mio. EUR, Endbestand 1,00 EUR."
- FALSCH: "Betriebs- und Geschaeftsausstattung stabil bei 39,9 Mio. EUR." (verwechselt Zugaenge mit Bestand!)

VOLLSTAENDIGKEIT (KRITISCH):
- Erfasse ALLE Zeilen der Tabelle, nicht nur die auffaelligsten.
- Wenn die Tabelle Zwischensummen und eine Gesamtsumme hat, nenne BEIDE.
- Fehlende Zeilen = fehlerhafte Beschreibung.

BILANZ-WARNUNG (KRITISCH bei Finanztabellen):
- Unterscheide IMMER zwischen ABSCHNITTS-Zwischensummen und der GESAMT-Summe.
- "Anlagevermoegen" und "Umlaufvermoegen" sind ABSCHNITTE (Zwischensummen).
- "Bilanzsumme", "Bilanzsumme Aktiva", "Bilanzsumme Passiva", "Gesamtsumme" oder "Summe Aktiva/Passiva" ist das GESAMTERGEBNIS.
- Die LETZTE Summenzeile der Tabelle ist fast immer die Bilanzsumme, NICHT ein Abschnittsname.
- FALSCH: "Gesamtsumme des Umlaufvermoegens betraegt X EUR" wenn X die Bilanzsumme ist.
- RICHTIG: "Bilanzsumme Aktiva betraegt X EUR" oder "Gesamtsumme Aktiva betraegt X EUR".
- Wenn die Tabelle NUR Umlaufvermoegen zeigt (kein Anlagevermoegen-Abschnitt), dann ist "Umlaufvermoegen" korrekt.
- Wenn BEIDE Abschnitte vorhanden sind, ist die letzte Summe die Bilanzsumme.

REGELN:
- alt_text: "Tabelle – " + Thema + Kernaussage (basierend auf den RICHTIGEN Endwerten). Max 250 Zeichen.
- GESAMTSUMME ZUERST: Wenn die Tabelle eine Bilanzsumme, Gesamtsumme oder Endsumme hat, nenne sie ZUERST im alt_text und am ANFANG der langbeschreibung. Sie ist die wichtigste Zahl und darf NICHT durch Token-Limits abgeschnitten werden.
- langbeschreibung: Gesamtsumme → Spaltenkoepfe → ALLE Zeilen mit korrekter Spaltenzuordnung → Spitzenwerte, Tiefstwerte, auffaellige Muster. Nur bei sehr kleinen Tabellen (max 4x4) alle Werte auflisten. KEINE Markdown-Tabellen im JSON! Max 1500 Zeichen.
- OCR-Text ([OCR-Text im Bild]) ist primaere Quelle fuer Zellinhalte und Zahlen.
- Einheiten (%, EUR, Mio.) penibel uebernehmen.
- ANTI-HALLUZINATION: Keine Werte erfinden. Bei unleserlichen Zellen: "Werte teilweise nicht lesbar".
- Sprache: Professionelles Deutsch, auch wenn Quelldaten englisch sind.
- KONTEXT-WARNUNG: Der Kontext ist Text NEBEN der Tabelle, nicht Teil der Tabelle selbst.

Antworte NUR mit diesem JSON:
{{"alt_text": "...", "langbeschreibung": "..."}}

Kontext: {context}""",

    "karte": """Du bist ein Experte fuer barrierefreie Bildbeschreibungen nach WCAG 2.2.
Dieses Bild ist eine KARTE (Landkarte, Stadtplan, Lageplan o.ae.).

DEIN AUFTRAG: Vermittle die RAEUMLICHE ERKENNTNIS. Was zeigt die Karte und was ist die geografische Kernaussage? Wo konzentrieren sich Markierungen? Welches Muster ist erkennbar?

EVIDENZ-BASIERTE IDENTIFIKATION:
- Ortsnamen, Legenden und Beschriftungen NUR nennen wenn im Bild LESBAR oder aus OCR-Text.
- Wenn spezifische Standorte markiert und beschriftet sind: ALLE aufzaehlen – das sind die Kerninformationen.
- Hintergrund-Ortsnamen (Staedte die nur zur Orientierung da sind) NICHT erschoepfend auflisten.

REGELN:
- alt_text: "Karte – " + Gebiet + Hauptthema + raeumliche Kernaussage. Max 350 Zeichen.
- langbeschreibung: Markierte Standorte vollstaendig auflisten. Legende erklaeren. Raeumliche Verteilung beschreiben. Max 1500 Zeichen.
- Originale Ortsnamen beibehalten (englische, franzoesische etc.).
- ANTI-HALLUZINATION: Keine Orte oder Routen erfinden. Bei verschwommenen Details: "Details teilweise nicht lesbar".
- Sprache: Deutsch.

Antworte NUR mit diesem JSON:
{{"alt_text": "...", "langbeschreibung": "..."}}

Kontext: {context}""",

    "logo": """Du bist ein Experte fuer barrierefreie Bildbeschreibungen nach WCAG 2.2.
Dieses Bild ist ein LOGO.

DEIN AUFTRAG: Der blinde Nutzer muss sofort wissen, welche Organisation oder Marke das ist. Sonst nichts. Kein visuelles Design, keine Farben, keine Formen.

EVIDENZ-BASIERTE IDENTIFIKATION:
- STUFE 1: Name ist als Text im Logo LESBAR → direkt nennen.
- STUFE 2: Logo ist ein weltweit eindeutiges Symbol (z.B. Apfel mit Biss = Apple, Stern im Kreis = Mercedes-Benz) UND der Kontext stuetzt die Identifikation → benennen.
- STUFE 3: Logo nicht identifizierbar → "Logo – Text nicht lesbar" oder "Logo eines nicht identifizierbaren Unternehmens".

LIZENZ- UND ZERTIFIZIERUNGSLOGOS:
- Creative Commons Logos: Exakt benennen mit Lizenztyp. Z.B. "Logo Creative Commons BY-ND" oder "Logo Creative Commons BY-SA 4.0".
- Zertifizierungssiegel (Bio, Fair Trade, TUeV etc.): Name des Siegels + ggf. Standard.
- Diese sind NICHT dekorativ – sie tragen rechtliche oder qualitaetsbezogene Information.

REGELN:
- Format: "Logo " + Name. Optional + Slogan wenn lesbar. NICHTS weiter.
- KEINE visuelle Beschreibung. Keine Wappen, Tiere, Formen, Farben.
- Max 1 Satz, 30-80 Zeichen.
- Verlinkte Logos: Wenn [Link-Ziel] im Kontext → "Logo [Name] – Link zur Startseite" o.ae.
- Sprache: Deutsch. Eigennamen und Slogans im Original.

Antworte NUR mit diesem JSON:
{{"alt_text": "...", "langbeschreibung": ""}}

Kontext: {context}""",

    "screenshot": """Du bist ein Experte fuer barrierefreie Bildbeschreibungen nach WCAG 2.2.
Dieses Bild ist ein SCREENSHOT einer Software, App oder Website.

DEIN AUFTRAG: Beschreibe den ZUSTAND und die FUNKTION, nicht das Layout. Was sieht der Nutzer inhaltlich? Welche Information oder Aktion steht im Fokus? Ignoriere dekorative UI-Elemente wie Rahmen, Schatten und Hintergrundfarben.

EVIDENZ-BASIERTE IDENTIFIKATION:
- Software/Website benennen wenn Name oder Logo LESBAR ist oder der Kontext es eindeutig sagt.
- Sichtbare Texte (Menuepunkte, Ueberschriften, Meldungen) wortwoertlich zitieren wenn relevant.
- OCR-Text als primaere Quelle nutzen wenn vorhanden.

REGELN:
- KEIN Praefix "Screenshot – ". Beginne direkt mit der Anwendung und dem Fokus.
- Beschreibe den ZENTRALEN Bereich – nicht jedes Icon in der Toolbar.
- ANTI-HALLUZINATION: Beschreibe NUR was auf dem Bildschirm zu sehen ist. Erfinde keine Nutzeraktionen.
- 2-4 Saetze, natuerliches Deutsch. UI-Begriffe duerfen im Original bleiben.

Antworte NUR mit diesem JSON:
{{"alt_text": "...", "langbeschreibung": "..."}}

alt_text: Max 350 Zeichen. Anwendung + Zustand + Fokus.
langbeschreibung: Weitere sichtbare Inhalte, Menuepunkte, Statusmeldungen – nur wenn informationsreich. Max 1000 Zeichen. Leer wenn alt_text ausreicht.

Kontext: {context}""",

    "infografik": """Du bist ein Experte fuer barrierefreie Bildbeschreibungen nach WCAG 2.2.
Dieses Bild ist eine INFOGRAFIK.
Bildgroesse: {width}x{height} Pixel

{thumbnail_block_infografik}

DEIN AUFTRAG: Uebersetze die visuell dargestellte WISSENSSTRUKTUR in Text. Was erklaert diese Infografik? Welcher Prozess, welche Hierarchie, welche Fakten werden vermittelt? Beschreibe die LOGIK, nicht das Layout.

BEISPIEL:
- SCHLECHT: "Infografik mit 5 Kaesten und Pfeilen die nach rechts zeigen."
- GUT: "Infografik – Der Gesetzgebungsprozess in fuenf Schritten: Entwurf, Ausschussberatung, erste Lesung, zweite Lesung, Verkuendung."

REGELN:
- alt_text: "Infografik – " + Hauptthema + zentrale Kernaussage. Max 350 Zeichen.
- langbeschreibung: Die inhaltlichen Stationen oder Fakten in logischer Reihenfolge. Beschreibe Beziehungen ("A fuehrt zu B", "X umfasst Y"), NICHT visuelles Layout ("oben links steht", "ein Pfeil zeigt auf"). Fliesstext, max 1500 Zeichen.
- OCR-Text als primaere Quelle nutzen.
- ANTI-HALLUZINATION VERSCHAERFT: Beschreibe NUR Inhalte die du im Bild LESEN oder ERKENNEN kannst. Der Seitenkontext verraet dir das Themenfeld, aber NICHT die konkreten Inhalte der Infografik. Wenn du Zahlen, Begriffe oder Zusammenhaenge nicht im Bild siehst, nenne sie NICHT.
- Sprache: Deutsch. Fachbegriffe aus dem Bild exakt uebernehmen.

Antworte NUR mit diesem JSON:
{{"alt_text": "...", "langbeschreibung": "..."}}

Kontext: {context}""",

    "strukturformel": """Du bist ein Experte fuer barrierefreie Bildbeschreibungen nach WCAG 2.2.
Dieses Bild zeigt eine CHEMISCHE STRUKTURFORMEL, ein Molekuelmodell oder eine Reaktionsgleichung.

DEIN AUFTRAG: Chemie verlangt Praezision UND Identifikation. Wenn die Struktur eindeutig ist, BENENNE das Molekuel – das ist die wichtigste Information fuer den Nutzer.

EVIDENZ-BASIERTE IDENTIFIKATION:
- STUFE 1: Name oder Summenformel steht als Text IM BILD oder im OCR-Text → direkt nennen.
- STUFE 2: Die Struktur ist chemisch EINDEUTIG identifizierbar (z.B. 4 C-Atome in verzweigter Kette mit korrekter Valenz = Isobutan/2-Methylpropan; Benzolring mit OH-Gruppe = Phenol) → benennen mit IUPAC-Name und Trivialname.
- STUFE 3: Struktur ist komplex oder mehrdeutig → nur beschreiben was sichtbar ist ("Strukturformel eines Molekuels mit aromatischem Ring und zwei Substituenten").

BEISPIELE:
- "Strukturformel – Isobutan (2-Methylpropan, C4H10). Verzweigte Kohlenstoffkette mit einem zentralen Kohlenstoffatom und drei Methylgruppen."
- "Reaktionsgleichung – Veresterung von Essigsaeure mit Ethanol zu Ethylacetat und Wasser unter Saeurekatalyse."

REGELN:
- alt_text: "Strukturformel – " oder "Reaktionsgleichung – " + Name + Summenformel. Max 250 Zeichen.
- langbeschreibung: Grundgeruest, Atome, Bindungen, funktionelle Gruppen. Bei Reaktionen: Edukte → Bedingungen → Produkte. Max 800 Zeichen Fliesstext.
- ANTI-HALLUZINATION: Erfinde KEINE Atome, Stereochemie oder Namen. Bei Unsicherheit: "Struktur nicht eindeutig identifizierbar" und nur visuell beschreiben.
- KONTEXT-WARNUNG: Der Kontext ist Text NEBEN dem Bild. Schliesse nicht blind vom Kontext auf das Molekuel wenn das Bild etwas anderes zeigt.
- Sprache: Fachlich korrektes Deutsch.

Antworte NUR mit diesem JSON:
{{"alt_text": "...", "langbeschreibung": "..."}}

Kontext: {context}""",

    "icon": """Du bist ein Experte fuer barrierefreie Bildbeschreibungen nach WCAG 2.2.
Dieses Bild ist ein ICON – ein kleines funktionales Symbol.

DEIN AUFTRAG: Beschreibe die FUNKTION, nicht das Aussehen. Was BEWIRKT dieses Icon? Welche Aktion loest es aus? Das ist alles was zaehlt.

BEISPIELE:
- Lupe → "Suche"
- Drei horizontale Striche → "Hauptmenue oeffnen"
- Einkaufswagen → "Warenkorb"
- Briefumschlag → "E-Mail senden"
- Wenn [Link-Ziel] vorhanden: Lupe mit Link zu /search → "Zur Suchseite"

REGELN:
- alt_text: Nur die Funktion. Max 50 Zeichen. Kein Praefix.
- langbeschreibung: Immer leer.
- Verlinkte Icons: Wenn [Link-Ziel] im Kontext → beschreibe wohin der Link fuehrt.
- Sprache: Deutsch. Technische UI-Begriffe sind erlaubt.
- DEKORATIV-CHECK: Rein dekorative Icons (Schmuckelemente ohne Funktion) → alt_text leer setzen.

Antworte NUR mit diesem JSON:
{{"alt_text": "...", "langbeschreibung": ""}}

Kontext: {context}""",

    "funktional": """Du bist ein Experte fuer barrierefreie Bildbeschreibungen nach WCAG 2.2.
Dieses Bild ist ein FUNKTIONALES UI-ELEMENT (Navigation, Paginierung, Button o.ae.).

DEIN AUFTRAG: Beschreibe die FUNKTION, nicht das Aussehen. Ein Screenreader-Nutzer muss wissen: Was passiert wenn ich dieses Element aktiviere? Was ist der aktuelle Zustand?

REGELN:
- Beschreibe die Funktion: "Naechste Seite", "Zurueck zur Uebersicht", "Menue oeffnen"
- Wenn ein Zustand sichtbar ist: "Keine vorherige Seite" (ausgegraut/deaktiviert), "Seite 3 von 5"
- KEIN visuelles Design beschreiben ("grauer Pfeil nach links")
- KURZ: Funktionale Alt-Texte sind typischerweise 2-6 Woerter
- langbeschreibung ist bei funktionalen Elementen IMMER leer

Bestehender Alt-Text: {original_alt}
Kontext: {context}

Antworte NUR mit diesem JSON:
{{"alt_text": "...", "langbeschreibung": ""}}""",

    "dekorativ": """Du bist ein Experte fuer barrierefreie Bildbeschreibungen nach WCAG 2.2.
Dieses Bild wurde als DEKORATIV vorklassifiziert. Deine Aufgabe: FINALE PRUEFUNG.

DEIN AUFTRAG: Schuetze blinde Nutzer vor digitalem Muell. Dekorative Bilder verschwenden ihre Zeit. ABER: Uebersehe keine echte Information. Pruefe rigoros.

ENTSCHEIDUNGSLOGIK:
1. Ist TEXT im Bild lesbar? → NIEMALS dekorativ. Beschreibe den Text.
2. Sind Personen, Objekte oder Daten erkennbar? → NICHT dekorativ. Beschreibe was zu sehen ist.
3. Ist es ein rein abstraktes Muster, Farbverlauf, Trennlinie oder Schmuckelement? → Dekorativ.

REGELN:
- Dekorativ: ist_dekorativ=true, alt_text="" (komplett leer)
- Nicht dekorativ: ist_dekorativ=false, alt_text mit kurzer Beschreibung (1-2 Saetze)
- ANTI-HALLUZINATION: Interpretiere NICHTS in abstrakte Formen hinein. Ein blauer Strich ist ein Strich, kein Fluss.
- KONTEXT-WARNUNG: Langer Kontexttext macht ein reines Hintergrundbild NICHT informativ.

Antworte NUR mit diesem JSON:
{{"alt_text": "...", "langbeschreibung": "", "ist_dekorativ": true/false, "konfidenz": "hoch|mittel|niedrig"}}

Kontext: {context}""",

}

# ─── Spezialisierte Fallback-Prompts (Qwen) ──────────────────

SPECIALIZED_FALLBACK_PROMPTS = {

    "foto": """/no_think
Du bist ein Experte fuer barrierefreie Bildbeschreibungen nach WCAG 2.2 und BITV.
Dieses Bild ist ein FOTO. Vermittle die BEDEUTUNG, nicht nur das Aussehen.

Antworte NUR mit diesem JSON:
{{"bildtyp": "foto", "alt_text": "...", "langbeschreibung": "...", "ist_dekorativ": false, "konfidenz": "hoch|mittel|niedrig"}}

EVIDENZ-REGELN:
- Text/Logo im Bild LESBAR → nennen (z.B. Firmenlogo, Schild, Inschrift)
- Lesbarer Text + Allgemeinwissen = sichere Identifikation → benennen (z.B. "EQUAL JUSTICE UNDER LAW" → Supreme Court)
- Kein Text, kein Logo → allgemein beschreiben ("ein Gebaeude", "eine Person")
- Personen namentlich NUR aus Kontext oder lesbarem Namensschild. Kein Alter.
- ANTI-REDUNDANZ: Wiederhole keine beschreibenden Details aus dem Kontext. Aber Namen und Identitaeten IMMER nennen.
- Unterschriften: Verwende den GEDRUCKTEN Namen neben/unter der Unterschrift. NIEMALS Handschrift entziffern.
- Farben NUR wenn informationstragend.

alt_text: Max 250 Zeichen, Kernaussage. KEIN Praefix "Foto – ".
langbeschreibung: Ergaenzende Details, max 1000 Zeichen. Leer wenn nicht noetig.
Sprache: Deutsch.

Kontext: {context}""",

    "diagramm": """/no_think
Du bist ein Experte fuer barrierefreie Bildbeschreibungen nach WCAG 2.2 und BITV.
Dieses Bild ist ein DIAGRAMM. Vermittle die ERKENNTNIS – den Trend, das Muster, die Kernaussage.

Antworte NUR mit diesem JSON:
{{"bildtyp": "diagramm", "alt_text": "...", "langbeschreibung": "...", "ist_dekorativ": false, "konfidenz": "hoch|mittel|niedrig"}}

INVENTAR-PFLICHT – VOR jeder Interpretation:
- Zaehle ALLE Kategorien auf der X-Achse. Wenn du 4 siehst, muessen alle 4 in die Beschreibung.
- Zaehle ALLE Legendeneintraege. Alle muessen erwaehnt werden.
- Nenne den Diagrammtitel wenn vorhanden.

TREND-VALIDIERUNG:
- Vergleiche JEDEN Datenpunkt paarweise: Jahr1→Jahr2, Jahr2→Jahr3, Jahr1→Jahr3.
- VERBOTEN: "kontinuierlich steigend" ohne alle Punkte geprueft zu haben.
- Pruefe jede Kategorie EINZELN. Nicht von einer auf andere schliessen.
- ERFINDE KEINE visuellen Maengel ("unleserlich", "ueberlappend") – wenn die Y-Achse lesbar ist, sag das.

INSIGHT-FIRST:
- Erster Satz = Diagrammtyp + Titel + Kernaussage/Trend.
- BEISPIEL: "Balkendiagramm – Umsatzentwicklung in Deutschland: Vier Kategorien (Software, Hardware, Services, Mobile) im Vergleich 2021-2023. Hardware erreicht 2022 den Hoechstwert und faellt 2023 zurueck."
- Nenne NUR Zahlen die du KLAR LESEN kannst. Keine Werte erfinden.

alt_text: Diagrammtyp + Titel + Kernaussage ueber ALLE Kategorien, max 350 Zeichen.
langbeschreibung: ALLE Kategorien und Legendeneintraege. Pro Kategorie die lesbaren Werte. Achsen, Hoechst-/Tiefstwerte. Max 1500 Zeichen. Keine Markdown-Tabellen.
Sprache: Deutsch.

Kontext: {context}""",

    "tabelle": """/no_think
Du bist ein Experte fuer barrierefreie Bildbeschreibungen nach WCAG 2.2 und BITV.
Dieses Bild zeigt eine TABELLE. Vermittle die KERNAUSSAGE, nicht nur die Struktur.

Antworte NUR mit diesem JSON:
{{"bildtyp": "tabelle", "alt_text": "...", "langbeschreibung": "...", "ist_dekorativ": false, "konfidenz": "hoch|mittel|niedrig"}}

SPALTEN-ZUORDNUNG (KRITISCH):
- Lies ZUERST alle Spaltenkoepfe. Ordne JEDEN Wert seiner Spalte zu.
- Verwechsle NIEMALS Zugaenge/Abschreibungen/Veraenderungen mit Anfangs- oder Endbestaenden.
- Bei Buchhaltungstabellen: Anfangs- und Endbestand sind die entscheidenden Werte.
- Erfasse ALLE Zeilen, nicht nur die groessten Werte. Fehlende Zeilen = fehlerhafte Beschreibung.
- Nenne Zwischensummen UND Gesamtsumme wenn vorhanden.

REGELN:
- alt_text: "Tabelle – " + Thema + Kernaussage (basierend auf RICHTIGEN Endwerten). Max 250 Zeichen.
- GESAMTSUMME ZUERST: Bilanzsumme/Gesamtsumme/Endsumme ZUERST im alt_text und am Anfang der langbeschreibung nennen. Sie darf nicht abgeschnitten werden.
- langbeschreibung: Gesamtsumme → Spaltenkoepfe → ALLE Zeilen mit korrekter Zuordnung → Spitzen-/Tiefstwerte. Fliesstext, max 1500 Zeichen. KEINE Markdown-Tabellen.
- Einheiten exakt uebernehmen.
- Keine Werte erfinden. Bei unleserlichen Zellen: "teilweise nicht lesbar".
- Sprache: Deutsch.

Kontext: {context}""",

    "karte": """/no_think
Du bist ein Experte fuer barrierefreie Bildbeschreibungen nach WCAG 2.2 und BITV.
Dieses Bild ist eine KARTE. Vermittle die raeumliche Information.

Antworte NUR mit diesem JSON:
{{"bildtyp": "karte", "alt_text": "...", "langbeschreibung": "...", "ist_dekorativ": false, "konfidenz": "hoch|mittel|niedrig"}}

REGELN:
- alt_text: "Karte – " + Gebiet + Hauptthema. Max 350 Zeichen.
- langbeschreibung: Markierte Standorte auflisten, Legende erklaeren, raeumliche Verteilung. Max 1500 Zeichen.
- Ortsnamen NUR wenn lesbar. Keine Orte erfinden.
- Originale Ortsnamen beibehalten.
- Sprache: Deutsch.

Kontext: {context}""",

    "logo": """/no_think
Dieses Bild ist ein LOGO. Nenne NUR den Namen der Organisation oder Marke.

Antworte NUR mit diesem JSON:
{{"bildtyp": "logo", "alt_text": "...", "ist_dekorativ": false, "konfidenz": "hoch|mittel|niedrig"}}

REGELN:
- Format: "Logo " + Name. Optional Slogan wenn lesbar.
- KEINE visuelle Beschreibung (keine Formen, Farben, Tiere, Wappen).
- Max 80 Zeichen.
- Name nicht lesbar → "Logo – Text nicht lesbar".
- Verlinkte Logos mit [Link-Ziel] → "Logo [Name] – Link zur Startseite".
- Sprache: Deutsch. Eigennamen im Original.

Kontext: {context}""",

    "screenshot": """/no_think
Du bist ein Experte fuer barrierefreie Bildbeschreibungen nach WCAG 2.2 und BITV.
Dieses Bild ist ein SCREENSHOT. Beschreibe FUNKTION und ZUSTAND, nicht das Layout.

Antworte NUR mit diesem JSON:
{{"bildtyp": "screenshot", "alt_text": "...", "langbeschreibung": "...", "ist_dekorativ": false, "konfidenz": "hoch|mittel|niedrig"}}

REGELN:
- KEIN Praefix "Screenshot – ". Direkt mit Anwendung und Fokus starten.
- Software/Website benennen wenn Name oder Logo LESBAR ist.
- Sichtbare Texte (Ueberschriften, Meldungen) zitieren wenn relevant.
- Nur den zentralen Bereich beschreiben, nicht jedes Menue-Icon.
- Keine Nutzeraktionen erfinden.
- alt_text: Max 350 Zeichen. langbeschreibung: Max 1000 Zeichen, leer wenn unnoetig.
- Sprache: Deutsch. UI-Begriffe im Original erlaubt.

Kontext: {context}""",

    "infografik": """/no_think
Du bist ein Experte fuer barrierefreie Bildbeschreibungen nach WCAG 2.2 und BITV.
Dieses Bild ist eine INFOGRAFIK. Vermittle die WISSENSSTRUKTUR – Prozesse, Fakten, Zusammenhaenge.

Antworte NUR mit diesem JSON:
{{"bildtyp": "infografik", "alt_text": "...", "langbeschreibung": "...", "ist_dekorativ": false, "konfidenz": "hoch|mittel|niedrig"}}

REGELN:
- alt_text: "Infografik – " + Hauptthema + Kernaussage. Max 350 Zeichen.
- langbeschreibung: Inhaltliche Stationen in logischer Reihenfolge. NICHT das Layout beschreiben. Max 1500 Zeichen.
- Fachbegriffe exakt uebernehmen.
- Keine Zusammenhaenge erfinden.
- Sprache: Deutsch.

Kontext: {context}""",

    "strukturformel": """/no_think
Du bist ein Experte fuer barrierefreie Bildbeschreibungen nach WCAG 2.2 und BITV.
Dieses Bild zeigt eine CHEMISCHE STRUKTURFORMEL oder Reaktionsgleichung.

Antworte NUR mit diesem JSON:
{{"bildtyp": "strukturformel", "alt_text": "...", "langbeschreibung": "...", "ist_dekorativ": false, "konfidenz": "hoch|mittel|niedrig"}}

EVIDENZ-REGELN:
- Name/Summenformel im Bild LESBAR → direkt nennen.
- Struktur chemisch EINDEUTIG (z.B. 4 C-Atome verzweigt = Isobutan) → benennen mit Name und Formel.
- Struktur NICHT eindeutig → nur beschreiben: "Strukturformel eines Molekuels mit Benzolring und Seitenkette".
- Erfinde KEINE Atome, Stereochemie oder IUPAC-Namen.

alt_text: "Strukturformel – " + Name + Formel. Max 250 Zeichen.
langbeschreibung: Grundgeruest, Atome, Bindungen, funktionelle Gruppen. Max 800 Zeichen.
Sprache: Fachlich korrektes Deutsch.

Kontext: {context}""",

    "icon": """/no_think
Dieses Bild ist ein ICON – ein funktionales Symbol.

Antworte NUR mit diesem JSON:
{{"bildtyp": "icon", "alt_text": "...", "ist_dekorativ": false, "konfidenz": "hoch|mittel|niedrig"}}

REGELN:
- Beschreibe NUR die FUNKTION. "Suche", "Hauptmenue", "Warenkorb", "E-Mail senden".
- KEIN visuelles Aussehen. KEIN Praefix.
- Verlinkte Icons mit [Link-Ziel] → wohin der Link fuehrt.
- Rein dekorative Icons → alt_text leer, ist_dekorativ=true.
- Max 50 Zeichen. Deutsch.

Kontext: {context}""",

    "funktional": """/no_think
Dieses Bild ist ein FUNKTIONALES UI-ELEMENT (Navigation, Paginierung, Button).

Antworte NUR mit diesem JSON:
{{"bildtyp": "funktional", "alt_text": "...", "ist_dekorativ": false, "konfidenz": "hoch|mittel|niedrig"}}

REGELN:
- Beschreibe NUR die FUNKTION: "Naechste Seite", "Zurueck", "Menue oeffnen".
- Wenn ein Zustand sichtbar ist: "Keine vorherige Seite" (deaktiviert).
- KEIN visuelles Design ("grauer Pfeil").
- KURZ: 2-6 Woerter.
- Bestehender Alt-Text: {original_alt}

Kontext: {context}""",

    "dekorativ": """/no_think
Dieses Bild wurde als DEKORATIV vorklassifiziert. Pruefe FINAL:

Antworte NUR mit diesem JSON:
{{"bildtyp": "dekorativ", "alt_text": "...", "ist_dekorativ": true/false, "konfidenz": "hoch|mittel|niedrig"}}

ENTSCHEIDUNG:
- Text im Bild lesbar? → NIEMALS dekorativ. Text beschreiben.
- Personen, Objekte, Daten erkennbar? → NICHT dekorativ. Kurz beschreiben.
- Rein abstraktes Muster, Farbverlauf, Trennlinie? → ist_dekorativ=true, alt_text=""
- Nichts in abstrakte Formen hineininterpretieren.

Kontext: {context}""",

}

# ─── Legacy General-Prompt (fuer qwen_only Modus und wenn kein Typ bekannt) ─
GENERAL_PROMPT = """/no_think
Du bist ein Experte fuer barrierefreie Bildbeschreibungen nach WCAG 2.2 und BITV.
ZIEL: Blinde Nutzer erhalten die GLEICHE ERKENNTNIS wie Sehende – nicht nur eine Pixelbeschreibung.

Antworte NUR mit diesem exakten JSON:
{{"bildtyp": "foto|diagramm|tabelle|screenshot|icon|logo|karte|dekorativ|strukturformel|infografik", "alt_text": "...", "langbeschreibung": "...", "ist_dekorativ": true/false, "konfidenz": "hoch|mittel|niedrig"}}

FORMAT-REGELN:
- Bei Diagrammen, Karten, Tabellen, Infografiken, Strukturformeln: Typ als Praefix (z.B. "Balkendiagramm – ...").
- Bei Fotos, Screenshots, Logos: KEIN Praefix, direkt mit Beschreibung starten.
- langbeschreibung: Nur bei Diagrammen, Tabellen, Karten, Infografiken, Strukturformeln und komplexen Screenshots. Sonst leer.
- Dekorativ: ist_dekorativ=true, alt_text="" und langbeschreibung=""

EVIDENZ-BASIERTE IDENTIFIKATION:
- Text/Logo im Bild KLAR LESBAR → direkt nennen.
- Lesbarer Text + Allgemeinwissen = sichere Identifikation → benennen. (Inschrift → Gebaeude identifizieren. Logo → Marke benennen. Eindeutige Molekuelstruktur → chemischen Namen nennen.)
- Kein Text, kein Logo, nur visueller Eindruck → allgemein beschreiben. NIEMALS Marken, Namen oder Orte raten.

DIAGRAMM-SPEZIFISCH:
- ZUERST: Zaehle ALLE Kategorien und Legendeneintraege. Alle muessen erwaehnt werden.
- Diagrammtitel IMMER nennen wenn vorhanden.
- Trends NUR nach paarweisem Vergleich aller Datenpunkte. KEIN "kontinuierlich" ohne Pruefung.
- Erfinde KEINE visuellen Maengel.

TABELLEN-SPEZIFISCH:
- ZUERST: Lies alle Spaltenkoepfe. Ordne JEDEN Wert seiner Spalte zu.
- Verwechsle NIEMALS Zugaenge/Abschreibungen mit Bestands-/Endwerten.
- Erfasse ALLE Zeilen. Nenne Zwischen- und Gesamtsummen.

WEITERE REGELN:
1. Sprache: Deutsch. Natuerlich formuliert, nicht roboterhaft.
2. Kontext nutzen um das Gezeigte zu IDENTIFIZIEREN. Nichts erfinden was weder im Bild noch im Kontext steht.
3. Text im Bild lesbar → NIEMALS dekorativ.
4. Farben NUR wenn informationstragend.
5. ANTI-REDUNDANZ: Beschreibende Details nicht wiederholen. Aber Namen, Funktionen und Identitaeten aus dem Kontext IMMER im Alt-Text nennen.
6. Personen: Name/Funktion NUR aus Kontext oder lesbarem Schild. KEIN Alter.
7. Unterschriften: Verwende den GEDRUCKTEN Namen neben/unter der Unterschrift. NIEMALS Handschrift entziffern.
8. Verlinkte Bilder mit [Link-Ziel] → Link-FUNKTION beschreiben.
8. Logos: NUR den Namen. Keine visuelle Beschreibung.
9. Bei unleserlichen Werten: "teilweise nicht lesbar". Erfinde KEINE Zahlen.

Kontext: {context}"""

# ─── Types that Mistral should always handle in hybrid mode ───
MISTRAL_TYPES = {"diagramm", "tabelle", "karte", "strukturformel", "infografik"}

# Complex types that should include langbeschreibung
COMPLEX_TYPES = {"diagramm", "karte", "tabelle", "infografik", "strukturformel", "screenshot"}


def is_thumbnail(width: int, height: int) -> bool:
    """Check if an image is too small for detailed analysis."""
    return (width > 0 and height > 0 and
            (width < MIN_DETAIL_WIDTH or height < MIN_DETAIL_HEIGHT))


def validate_dekorativ(classification: dict, original_alt: str = "", width: int = 0, height: int = 0) -> str:
    """Safety net: Verify that 'dekorativ' classification is correct.
    Returns corrected bildtyp if misclassified.

    v2.2: Catches functional elements and license logos that Qwen misclassifies as decorative.
    """
    if classification.get("bildtyp") != "dekorativ":
        return classification["bildtyp"]

    alt_lower = (original_alt or "").strip().lower()

    # Check 1: Functional original alt-text → not decorative
    funktionale_keywords = [
        "seite", "navigation", "nächste", "naechste", "vorherige",
        "zurück", "zurueck", "weiter", "menü", "menue",
        "öffnen", "oeffnen", "schließen", "schliessen",
        "suche", "filter", "sortier"
    ]
    if any(kw in alt_lower for kw in funktionale_keywords):
        return "funktional"

    # Check 2: License/certification keywords → logo
    lizenz_keywords = [
        "creative commons", "cc by", "cc-by", "lizenz", "license",
        "copyright", "zertifizier", "siegel", "gütesiegel", "guetesiegel"
    ]
    if any(kw in alt_lower for kw in lizenz_keywords):
        return "logo"

    # Check 3: Low confidence → safer to treat as icon than empty alt
    if classification.get("konfidenz") == "niedrig":
        return "icon"

    return "dekorativ"

# Patterns for detecting image types from surrounding text
_TYPE_PATTERNS = {
    "diagramm": [
        r"(?i)\b(?:Diagramm|Balkendiagramm|Kreisdiagramm|Liniendiagramm|Saeulend|Tortendiagramm|Chart|Graph)\b",
        r"(?i)\b(?:Abb(?:ildung)?\.?\s*\d+\s*:\s*.*(?:Diagramm|Verteilung|Entwicklung|Verlauf|Vergleich))",
    ],
    "karte": [
        r"(?i)\b(?:Karte|Landkarte|Stadtplan|Lageplan|Map|Uebersichtskarte|Standortkarte)\b",
    ],
    "tabelle": [
        r"(?i)\b(?:Tabelle|Tab\.?\s*\d|Uebersicht\s*\d)\b",
    ],
    "infografik": [
        r"(?i)\b(?:Infografik|Schaubild|Uebersichtsgrafik)\b",
    ],
    "foto": [
        r"(?i)\b(?:Foto|Bild|Aufnahme|Portrait|Gruppenbild)\b",
    ],
    "logo": [
        r"(?i)\b(?:Logo|Wortmarke|Bildmarke|Firmenzeichen|Markenzeichen)\b",
    ],
    "screenshot": [
        r"(?i)\b(?:Screenshot|Bildschirmfoto|Bildschirmaufnahme|Screencapture)\b",
    ],
    "dekorativ": [
        r"(?i)\b(?:Schmuckbild|Dekoration|Zierbild|Hintergrundbild|Titelbild)\b",
    ],
    "strukturformel": [
        r"(?i)\b(?:Strukturformel|Molekuel|Summenformel|Reaktionsgleichung|Bindung|Atom|IUPAC)\b",
        r"(?i)\b(?:Methanol|Ethanol|Iodmethan|Kohlenwasserstoff|Wasserstoff|Sauerstoff|Stickstoff)\b",
    ],
    "icon": [
        r"(?i)\b(?:Icon|Symbol|Lupe|Hamburger|Menue-Icon|Warenkorb|Pfeil|Button-Icon)\b",
        r"(?i)\b(?:Suchicon|Menuesymbol|Navigationssymbol|Schaltflaeche|Toolbar-Icon)\b",
    ],
}


def get_classification_prompt(context_text: str = "", width: int = 0, height: int = 0, original_alt: str = "") -> str:
    """Return the classification prompt for Qwen (Stufe 1).
    v2.2: Now includes image dimensions and original alt-text for better classification.
    """
    context = context_text[:400] if context_text else "Kein Kontext."
    alt = original_alt.strip() if original_alt else "(kein Alt-Text vorhanden)"
    return CLASSIFICATION_PROMPT.format(
        context=context,
        width=width or "unbekannt",
        height=height or "unbekannt",
        original_alt=alt
    )


# v2.2.1: Postfilter – removes context interpretations from alt-texts
KONTEXT_PHRASEN = [
    r',?\s*symbolisiert\s+.*$',
    r',?\s*steht symbolisch für\s+.*$',
    r',?\s*repräsentiert\s+.*$',
    r',?\s*thematisch passend zu\s+.*$',
    r',?\s*(?:vermutlich\s+)?im Kontext (?:von|der|des)\s+.*$',
    r',?\s*(?:vermutlich\s+)?im Zusammenhang mit\s+.*$',
    r',?\s*passend zu den (?:Themen|Inhalten)\s+.*$',
    r',?\s*was auf .* hindeutet\.?$',
    r',?\s*typisch für\s+.*$',
    # v2.2.3: Erweiterte Anti-Raten-Regeln
    r',?\s*vermutlich\b.*$',
    r',?\s*wahrscheinlich\b.*$',
    r',?\s*möglicherweise\b.*$',
    r',?\s*Symbol für\s+.*$',
    r',?\s*im Rahmen (?:von|der|des|eines)\s+.*$',
]

def clean_alt_text(alt_text: str) -> str:
    """Entfernt Kontextinterpretationen am Satzende.
    Laeuft NACH jeder Alt-Text-Generierung, unabhaengig vom Pfad."""
    if not alt_text:
        return alt_text
    cleaned = alt_text
    for pattern in KONTEXT_PHRASEN:
        cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)
    cleaned = cleaned.rstrip(' ,.')
    if cleaned and not cleaned.endswith(')'):
        cleaned += '.'
    return cleaned if cleaned else alt_text


# v2.2.3: Hedge-Woerter auch INNERHALB des Satzes entfernen
HEDGE_PATTERNS = [
    (r'\bvermutlich\s+', ''),
    (r'\bwahrscheinlich\s+', ''),
    (r'\bmöglicherweise\s+', ''),
    (r'\boffenbar\s+', ''),
    (r'\banscheinend\s+', ''),
    (r'\bwohl\s+', ''),
]

def remove_hedge_words(alt_text: str) -> str:
    """Entfernt Unsicherheitswoerter aus Alt-Texten.
    v2.2.3: Alt-Text beschreibt was zu sehen ist, nicht was vermutet wird."""
    if not alt_text:
        return alt_text
    cleaned = alt_text
    for pattern, replacement in HEDGE_PATTERNS:
        cleaned = re.sub(pattern, replacement, cleaned, flags=re.IGNORECASE)
    # Doppelte Leerzeichen bereinigen
    cleaned = re.sub(r'  +', ' ', cleaned).strip()
    return cleaned if cleaned else alt_text


def _extract_link_from_context(context: str) -> str:
    """v2.2.3: Extract link target from scraper context string.
    Prefers [Link-Beschriftung] over [Link-Ziel] (human-readable)."""
    if not context:
        return ""
    import re as _re
    # Prefer readable label
    match = _re.search(r'\[Link-Beschriftung\]\s*(.+)', context)
    if match:
        label = match.group(1).strip()
        if label and len(label) > 2:
            return label
    # Fallback to URL
    match = _re.search(r'\[Link-Ziel\]\s*(.+)', context)
    if match:
        return match.group(1).strip()
    return ""


def extract_link_target(original_alt: str) -> str:
    """Extract link target from original alt-text if present.
    Returns the link target string or empty string.
    Patterns: '(verweist auf: ...)', '(Link: ...)', '(Link zu: ...)'
    """
    if not original_alt:
        return ""
    match = re.search(r'\(verweist auf:\s*(.+?)\)$', original_alt.strip())
    if match:
        return match.group(1).strip()
    match = re.search(r'\(Link(?:\s+zu)?:\s*(.+?)\)$', original_alt.strip())
    if match:
        return match.group(1).strip()
    return ""


def _get_thumbnail_block(width: int, height: int) -> str:
    """Return thumbnail warning block if image is too small for detail."""
    is_thumbnail = (width > 0 and height > 0 and
                    (width < MIN_DETAIL_WIDTH or height < MIN_DETAIL_HEIGHT))
    if not is_thumbnail:
        return ""
    return f"""ACHTUNG – THUMBNAIL-MODUS:
Dieses Bild ist nur {width}x{height} Pixel gross. Bei dieser Aufloesung kannst du KEINE feinen Details erkennen.
- Beschreibe NUR das grobe Motiv (z.B. "Landschaftsaufnahme", "Personengruppe", "Nahaufnahme von Lebensmitteln").
- Erfinde KEINE Details die bei dieser Groesse nicht erkennbar sind.
- KEINE Langbeschreibung generieren – setze langbeschreibung auf "".
- Wenn du nicht sicher bist was das Bild zeigt: Sag es ehrlich. "Kleines Vorschaubild, Motiv nicht eindeutig erkennbar" ist besser als eine erfundene Beschreibung."""


def _get_thumbnail_block_infografik(width: int, height: int) -> str:
    """Return thumbnail warning block for infographics."""
    is_thumbnail = (width > 0 and height > 0 and
                    (width < MIN_DETAIL_WIDTH or height < MIN_DETAIL_HEIGHT))
    if not is_thumbnail:
        return ""
    return f"""ACHTUNG – THUMBNAIL-MODUS:
Dieses Bild ist nur {width}x{height} Pixel gross. Bei dieser Aufloesung sind Details einer Infografik NICHT lesbar.
- Beschreibe NUR das erkennbare Gesamtlayout (z.B. "Kleine Vorschau einer Infografik").
- ERFINDE KEINE Inhalte die du nicht lesen kannst.
- Wenn OCR-Text vorhanden ist, nutze diesen. Wenn nicht: Ehrlich sagen dass Details nicht erkennbar sind.
- langbeschreibung auf "" setzen."""


def _get_verbesserung_block(original_alt: str, original_alt_brauchbar: bool) -> str:
    """Return improvement mode block if original alt-text is usable."""
    if not original_alt_brauchbar or not original_alt or not original_alt.strip():
        return ""
    alt = original_alt.strip()
    return f"""VERBESSERUNGSMODUS:
Fuer dieses Bild existiert bereits ein Alt-Text: "{alt}"
- Pruefe ob dieser Alt-Text korrekt und ausreichend ist.
- Wenn ja: Uebernimm ihn woertlich oder mit minimaler Verbesserung.
- Wenn nein: Verbessere ihn, aber behalte korrekte Informationen bei (besonders Link-Ziele und Funktionsbeschreibungen).
- VERSCHLECHTERE niemals einen guten bestehenden Alt-Text."""


def get_generation_prompt(bildtyp: str, context_text: str = "", width: int = 0, height: int = 0,
                          original_alt: str = "", original_alt_brauchbar: bool = False) -> str:
    """Return the specialized generation prompt for Mistral (Stufe 2).

    v2.2: Supports thumbnail mode, improvement mode, and functional elements.
    v2.2.1: Extracts link targets from original alt and adds as explicit context field.
    Falls back to GENERAL_PROMPT if the bildtyp has no specialized prompt.
    Context limit: 1200 chars for Mistral.
    """
    context = context_text[:1500] if context_text else "Kein Kontext."  # v2.2.3: raised from 1200 (page profile now max 300)

    # v2.2.3: Extract link target from original alt OR from scraper context
    link_target = extract_link_target(original_alt)
    if not link_target:
        # Fallback: extract from context (scraper adds [Link-Beschriftung] or [Link-Ziel])
        link_target = _extract_link_from_context(context)
    if link_target:
        context = f"[Link-Ziel dieses Bildes]: {link_target}\n{context}\n[WICHTIG – PFLICHT]: Dieses Bild ist ein Link. Der Alt-Text MUSS enden mit: (verweist auf: {link_target})"
    prompt_template = SPECIALIZED_GENERATION_PROMPTS.get(bildtyp)
    if prompt_template is None:
        return GENERAL_PROMPT.format(context=context)

    # Build format kwargs based on what the template expects
    fmt = {"context": context}

    # All prompts that have {width}/{height}
    if "{width}" in prompt_template:
        fmt["width"] = width or "unbekannt"
        fmt["height"] = height or "unbekannt"

    # Foto prompt: thumbnail + verbesserung blocks
    if "{thumbnail_block}" in prompt_template:
        fmt["thumbnail_block"] = _get_thumbnail_block(width, height)
    if "{verbesserung_block}" in prompt_template:
        fmt["verbesserung_block"] = _get_verbesserung_block(original_alt, original_alt_brauchbar)

    # Infografik prompt: thumbnail block
    if "{thumbnail_block_infografik}" in prompt_template:
        fmt["thumbnail_block_infografik"] = _get_thumbnail_block_infografik(width, height)

    # Funktional prompt: original_alt
    if "{original_alt}" in prompt_template:
        fmt["original_alt"] = original_alt.strip() if original_alt else "(kein Alt-Text vorhanden)"

    return prompt_template.format(**fmt)


def get_fallback_prompt(bildtyp: str, context_text: str = "", original_alt: str = "") -> str:
    """Return the specialized Qwen fallback prompt.

    v2.2: Supports original_alt for funktional prompts.
    Falls back to GENERAL_PROMPT if the bildtyp has no specialized fallback prompt.
    Context limit: 800 chars for Qwen.
    """
    context = context_text[:800] if context_text else "Kein Kontext."
    prompt_template = SPECIALIZED_FALLBACK_PROMPTS.get(bildtyp)
    if prompt_template is None:
        return GENERAL_PROMPT.format(context=context)
    fmt = {"context": context}
    if "{original_alt}" in prompt_template:
        fmt["original_alt"] = original_alt.strip() if original_alt else "(kein Alt-Text vorhanden)"
    return prompt_template.format(**fmt)


def get_prompt(image_type: str = None, context_text: str = "") -> str:
    """Legacy: Return GENERAL_PROMPT for Qwen-only/fallback mode."""
    context = context_text[:800] if context_text else "Kein Kontext."
    return GENERAL_PROMPT.format(context=context)


def should_use_mistral(bildtyp: str, konfidenz: str) -> bool:
    """Decide if Mistral should generate the alt-text based on pipeline mode."""
    if PIPELINE_MODE == "qwen_only":
        return False
    if PIPELINE_MODE == "mistral_primary":
        return True
    # hybrid mode
    if bildtyp in MISTRAL_TYPES:
        return True
    if konfidenz in ("niedrig", "mittel"):
        return True
    return False


def detect_type_from_context(context_text: str) -> str | None:
    """Analyze surrounding text for clues about the image type."""
    if not context_text:
        return None
    scores = {}
    for img_type, patterns in _TYPE_PATTERNS.items():
        score = 0
        for pattern in patterns:
            matches = re.findall(pattern, context_text)
            score += len(matches)
        if score > 0:
            scores[img_type] = score
    if not scores:
        return None
    return max(scores, key=scores.get)


def is_complex_type(image_type: str) -> bool:
    """Check if an image type should include a langbeschreibung field."""
    return image_type in COMPLEX_TYPES


def extract_page_profile(soup) -> str:
    """Extract a compact page profile from BeautifulSoup HTML.

    v2.2.3: Only title + meta description, no headings (max 300 chars).
    Headings come via image-specific context (nearest heading to each image).
    This prevents the page profile from consuming the context budget."""
    parts = []
    if soup.title and soup.title.string:
        parts.append(f"[Seitentitel] {soup.title.string.strip()[:100]}")
    meta_desc = soup.find("meta", attrs={"name": "description"})
    if meta_desc and meta_desc.get("content"):
        parts.append(f"[Meta-Beschreibung] {meta_desc['content'][:150]}")
    profile = "\n".join(parts) if parts else ""
    return profile[:300]

