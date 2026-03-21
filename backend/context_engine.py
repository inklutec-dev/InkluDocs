"""
Context Engine for InkluDocs Alt-Text Generation.
Version 2.0 – Evidenz-Stufenmodell (21.03.2026)

Pipeline:
  Stufe 1 (Qwen, lokal): Klassifikation (Bildtyp + dekorativ + konfidenz)
  Stufe 2 (Mistral, API): Alt-Text-Generierung mit spezialisierten Prompts
  Fallback: Qwen generiert wenn Mistral fehlschlaegt

Modi (PIPELINE_MODE Umgebungsvariable):
  mistral_primary: Qwen klassifiziert, Mistral generiert (Default, beste Qualitaet)
  hybrid: Qwen macht einfache Bilder, Mistral die komplexen (spart Kosten)
  qwen_only: Alles ueber Qwen (Fallback, niedrigste Kosten)
"""

import os
import re

PIPELINE_MODE = os.environ.get("PIPELINE_MODE", "mistral_primary")

# ─── Klassifikations-Prompt (Stufe 1, Qwen) ──────────────────
CLASSIFICATION_PROMPT = """/no_think
Klassifiziere dieses Bild. Antworte NUR mit diesem JSON:
{{"bildtyp": "foto|diagramm|tabelle|strukturformel|logo|icon|karte|screenshot|infografik|dekorativ", "ist_dekorativ": true|false, "konfidenz": "hoch|mittel|niedrig"}}

Regeln:
- dekorativ = rein abstrakte Formen, Farbverlaeufe, Trennlinien, Schmuckelemente ohne jede Information
- Wenn Text, Personen, Daten, Diagramme oder konkrete Objekte sichtbar sind: NICHT dekorativ
- icon = einzelnes kleines Symbol mit funktionaler Bedeutung (Lupe, Hamburger-Menue, Pfeil, Warenkorb)
- konfidenz: hoch = eindeutig, mittel = wahrscheinlich, niedrig = unklar

Kontext: {context}"""

# ─── Spezialisierte Generierungs-Prompts (Stufe 2, Mistral) ──

SPECIALIZED_GENERATION_PROMPTS = {

    "foto": """Du bist ein Experte fuer barrierefreie Bildbeschreibungen nach WCAG 2.2.
Dieses Bild ist ein FOTO.

DEIN AUFTRAG: Vermittle nicht was das Bild ZEIGT, sondern was es BEDEUTET. Was wuerde ein Sehender sofort denken und wissen? Genau das muss der blinde Nutzer erfahren.

EVIDENZ-BASIERTE IDENTIFIKATION:
- STUFE 1 (IMMER): Text, Namen, Logos die im Bild KLAR LESBAR sind → direkt nennen.
- STUFE 2 (ERLAUBT): Lesbarer Text oder eindeutiges Logo + Allgemeinwissen → benennen. Beispiel: Inschrift "EQUAL JUSTICE UNDER LAW" → "Supreme Court der USA". Mercedes-Stern sichtbar → "Mercedes-Benz". Aber Fahrzeugmodell NUR wenn als Text lesbar oder aus Kontext eindeutig.
- STUFE 3 (VERBOTEN): Kein Text, kein Logo, nur visueller Eindruck → allgemein beschreiben. "Ein industrielles Steuerungsmodul", NICHT "Siemens". "Eine Person", NICHT einen Namen raten.

REGELN:
- KEIN Praefix "Foto – ". Starte direkt mit der Erkenntnis.
- Personen: Name und Funktion NUR aus dem Kontext oder von lesbaren Namensschildern. KEIN Alter nennen.
- Gebaeude/Orte: Benennen wenn Evidenz vorhanden (Schild, Inschrift, Kontext). Sonst allgemein.
- Gruppen: Ungefaehre Anzahl, Anlass, Setting – was PASSIERT hier?
- Verlinkte Bilder: Wenn [Link-Ziel] im Kontext → beschreibe die FUNKTION des Links, nicht das Bild.
- ANTI-REDUNDANZ: Wiederhole NICHTS was der Kontext bereits sagt. Ergaenze, was nur das Bild zeigt.
- Farben NUR wenn sie Information tragen (Warnschilder, Signalfarben). Keine optischen Farben.
- Sprache: Natuerliches Deutsch. Kein Aufzaehlungsstil. Wie ein Mensch es einem anderen erzaehlen wuerde.

Antworte NUR mit diesem JSON:
{{"alt_text": "...", "langbeschreibung": "..."}}

alt_text: Max 250 Zeichen. Die Kernaussage – was ein Sehender sofort erkennt und denkt.
langbeschreibung: Ergaenzende Details die fuer tieferes Verstaendnis wichtig sind, max 1000 Zeichen. Leer lassen wenn der alt_text bereits alles Wesentliche enthaelt.

Kontext: {context}""",

    "diagramm": """Du bist ein Experte fuer barrierefreie Bildbeschreibungen nach WCAG 2.2.
Dieses Bild ist ein DIAGRAMM.

DEIN AUFTRAG – INSIGHT FIRST: Ein Sehender schaut auf ein Diagramm und erkennt sofort den TREND, das MUSTER, die KERNAUSSAGE. Genau DAS muss dein erster Satz sein. Nicht die Achsenbeschriftung. Nicht der Diagrammtyp. Sondern: Was sagt uns dieses Diagramm?

BEISPIELE FUER GUTE INSIGHTS:
- "Balkendiagramm – Der Umsatz im Bereich Services hat sich seit 2021 verdoppelt, waehrend Hardware ruecklaeufig ist – eine gegenlaeufige Entwicklung."
- "Kreisdiagramm – Fast die Haelfte der Buerokratie-Entlastung entfaellt auf ein einziges Gesetz, das Wachstumschancengesetz mit 39%."
- "Liniendiagramm – Nach stetigem Wachstum bis 2022 bricht der Trend 2023 abrupt ein."

EVIDENZ-BASIERTE IDENTIFIKATION:
- Nenne NUR Zahlen, Beschriftungen und Legenden die du KLAR LESEN kannst.
- Wenn Werte wegen verschluesselter Schrift oder niedriger Aufloesung nicht lesbar sind: "Werte teilweise nicht lesbar" – aber beschreibe trotzdem den SICHTBAREN Trend (steigend, fallend, gleichbleibend).
- Wenn OCR-Text bereitgestellt wird ([OCR-Text im Bild]), nutze diesen als primaere Datenquelle.

REGELN:
- alt_text: Diagrammtyp + Bindestrich + Kernaussage/Trend. 2-3 Saetze, max 350 Zeichen.
- langbeschreibung: Alle lesbaren Datenpunkte strukturiert auflisten. Achsenbeschriftungen, Legendenwerte, Anfangs-/Endwerte bei Zeitreihen. Hoechst- und Tiefstwerte benennen. Fliesstext oder strukturierte Liste, KEINE Markdown-Tabellen. Max 1500 Zeichen.
- Sprache: Deutsch. Insight zuerst, Details danach.
- ANTI-HALLUZINATION: Erfinde KEINE Zahlen. Wenn du einen Wert nicht sicher lesen kannst, lass ihn weg.

Antworte NUR mit diesem JSON:
{{"alt_text": "...", "langbeschreibung": "..."}}

Kontext: {context}""",

    "tabelle": """Du bist ein Experte fuer barrierefreie Bildbeschreibungen nach WCAG 2.2.
Dieses Bild zeigt eine TABELLE als Grafik.

DEIN AUFTRAG – INSIGHT FIRST: Was ist die ERKENNTNIS dieser Tabelle? Welchen Schluss zieht ein Sehender auf den ersten Blick? Fuehre damit. Dann die Daten.

BEISPIEL:
- SCHLECHT: "Tabelle mit 4 Spalten und 6 Zeilen zu Umsatzzahlen."
- GUT: "Tabelle – Produkt A dominiert mit 40% den Gesamtumsatz 2023, waehrend Produkt D unter 5% liegt."

REGELN:
- alt_text: "Tabelle – " + Thema + Kernaussage. Max 250 Zeichen.
- langbeschreibung: Spaltenkoepfe zuerst. Dann die wichtigsten Datenpunkte in Fliesstext. Spitzenwerte, Tiefstwerte, auffaellige Muster. Nur bei sehr kleinen Tabellen (max 4x4) alle Werte auflisten. KEINE Markdown-Tabellen im JSON! Max 1500 Zeichen.
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

DEIN AUFTRAG: Uebersetze die visuell dargestellte WISSENSSTRUKTUR in Text. Was erklaert diese Infografik? Welcher Prozess, welche Hierarchie, welche Fakten werden vermittelt? Beschreibe die LOGIK, nicht das Layout.

BEISPIEL:
- SCHLECHT: "Infografik mit 5 Kaesten und Pfeilen die nach rechts zeigen."
- GUT: "Infografik – Der Gesetzgebungsprozess in fuenf Schritten: Entwurf, Ausschussberatung, erste Lesung, zweite Lesung, Verkuendung."

REGELN:
- alt_text: "Infografik – " + Hauptthema + zentrale Kernaussage. Max 350 Zeichen.
- langbeschreibung: Die inhaltlichen Stationen oder Fakten in logischer Reihenfolge. Beschreibe Beziehungen ("A fuehrt zu B", "X umfasst Y"), NICHT visuelles Layout ("oben links steht", "ein Pfeil zeigt auf"). Fliesstext, max 1500 Zeichen.
- OCR-Text als primaere Quelle nutzen.
- ANTI-HALLUZINATION: Keine Zusammenhaenge erfinden die nicht im Bild stehen.
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
- ANTI-REDUNDANZ: Wiederhole NICHTS aus dem Kontext.
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

INSIGHT-FIRST:
- Erster Satz = Diagrammtyp + Kernaussage/Trend.
- BEISPIEL: "Balkendiagramm – Der Umsatz stieg von 2021 bis 2023 um 40%, mit dem staerksten Wachstum in 2022."
- Nenne NUR Zahlen die du KLAR LESEN kannst. Keine Werte erfinden.
- Bei unleserlichen Werten: "Werte teilweise nicht lesbar" – aber den sichtbaren Trend beschreiben.

alt_text: Diagrammtyp + Kernaussage, max 350 Zeichen.
langbeschreibung: Alle lesbaren Datenpunkte, Achsen, Legenden. Max 1500 Zeichen. Keine Markdown-Tabellen.
Sprache: Deutsch.

Kontext: {context}""",

    "tabelle": """/no_think
Du bist ein Experte fuer barrierefreie Bildbeschreibungen nach WCAG 2.2 und BITV.
Dieses Bild zeigt eine TABELLE. Vermittle die KERNAUSSAGE, nicht nur die Struktur.

Antworte NUR mit diesem JSON:
{{"bildtyp": "tabelle", "alt_text": "...", "langbeschreibung": "...", "ist_dekorativ": false, "konfidenz": "hoch|mittel|niedrig"}}

REGELN:
- alt_text: "Tabelle – " + Thema + eine klare Erkenntnis. Max 250 Zeichen.
- langbeschreibung: Spaltenkoepfe → wichtigste Datenpunkte → Spitzen-/Tiefstwerte. Fliesstext, max 1500 Zeichen. KEINE Markdown-Tabellen.
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

WEITERE REGELN:
1. Sprache: Deutsch. Natuerlich formuliert, nicht roboterhaft.
2. Kontext nutzen um das Gezeigte zu IDENTIFIZIEREN. Nichts erfinden was weder im Bild noch im Kontext steht.
3. Text im Bild lesbar → NIEMALS dekorativ.
4. Farben NUR wenn informationstragend.
5. ANTI-REDUNDANZ: Nichts wiederholen was der Kontext bereits sagt.
6. Personen: Name/Funktion NUR aus Kontext oder lesbarem Schild. KEIN Alter.
7. Verlinkte Bilder mit [Link-Ziel] → Link-FUNKTION beschreiben.
8. Diagramme/Tabellen: INSIGHT FIRST – Trend und Kernaussage zuerst, dann Details.
9. Logos: NUR den Namen. Keine visuelle Beschreibung.
10. Bei unleserlichen Werten: "teilweise nicht lesbar".

Kontext: {context}"""

# ─── Types that Mistral should always handle in hybrid mode ───
MISTRAL_TYPES = {"diagramm", "tabelle", "karte", "strukturformel", "infografik"}

# Complex types that should include langbeschreibung
COMPLEX_TYPES = {"diagramm", "karte", "tabelle", "infografik", "strukturformel", "screenshot"}

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


def get_classification_prompt(context_text: str = "") -> str:
    """Return the classification prompt for Qwen (Stufe 1)."""
    context = context_text[:400] if context_text else "Kein Kontext."
    return CLASSIFICATION_PROMPT.format(context=context)


def get_generation_prompt(bildtyp: str, context_text: str = "") -> str:
    """Return the specialized generation prompt for Mistral (Stufe 2).

    Falls back to GENERAL_PROMPT if the bildtyp has no specialized prompt.
    Context limit: 1200 chars for Mistral.
    """
    context = context_text[:1200] if context_text else "Kein Kontext."
    prompt_template = SPECIALIZED_GENERATION_PROMPTS.get(bildtyp)
    if prompt_template is None:
        return GENERAL_PROMPT.format(context=context)
    return prompt_template.format(context=context)


def get_fallback_prompt(bildtyp: str, context_text: str = "") -> str:
    """Return the specialized Qwen fallback prompt.

    Falls back to GENERAL_PROMPT if the bildtyp has no specialized fallback prompt.
    Context limit: 800 chars for Qwen.
    """
    context = context_text[:800] if context_text else "Kein Kontext."
    prompt_template = SPECIALIZED_FALLBACK_PROMPTS.get(bildtyp)
    if prompt_template is None:
        return GENERAL_PROMPT.format(context=context)
    return prompt_template.format(context=context)


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
    """Extract a page profile from BeautifulSoup HTML for better context."""
    parts = []
    if soup.title and soup.title.string:
        parts.append(f"[Seitentitel] {soup.title.string.strip()}")
    meta_desc = soup.find("meta", attrs={"name": "description"})
    if meta_desc and meta_desc.get("content"):
        parts.append(f"[Meta-Beschreibung] {meta_desc['content'][:200]}")
    headings = []
    for tag in ["h1", "h2", "h3"]:
        for h in soup.find_all(tag):
            text = h.get_text(strip=True)
            if text:
                headings.append(f"{tag.upper()}: {text}")
    if headings:
        parts.append(f"[Ueberschriften] {' | '.join(headings[:10])}")
    return "\n".join(parts) if parts else ""
