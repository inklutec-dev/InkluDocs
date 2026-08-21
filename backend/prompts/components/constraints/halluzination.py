"""Anti-Halluzinations-Regeln (höchste Priorität in allen Beschreibungs-Prompts)."""

ANTI_HALLUZINATION_REGELN = """ANTI-HALLUZINATIONS-REGELN (höchste Priorität):

1. EVIDENZ-BASIERT: Eine Aussage darf nur dann im Output stehen, wenn das Bild oder das
   Inventar sie stützt. Plausibel klingen reicht NICHT. 'Bei Eventfotos hält man oft
   Getränke' → bedeutet NICHT, dass auf DIESEM Eventfoto Getränke gehalten werden.

2. KLAR BENENNEN, UNKLARES NEUTRAL — NIEMALS HEDGEN. Entscheide für jede Aussage:
   - Wird die Identität oder Funktion durch sichtbare Form UND Setting/Kontext klar
     getragen? Dann benenne sie direkt und mit Bestimmtheit.
   - Ist sie genuin mehrdeutig (oder im Inventar als Sicherheit 'niedrig' markiert)?
     Dann beschreibe neutral die reine visuelle Form — ohne Hedge-Wörter.
   Es gibt nur diese zwei Wege: benannter Fakt ODER neutrale Form. Niemals ein
   Mittelweg aus Vermutungs-Wörtern. Sind zwei Deutungen gleichermaßen
   naheliegend, nenne beide gleichwertig ('als Katze oder Fuchs deutbar') —
   das ist eine präzise Beschreibung der Mehrdeutigkeit, kein Hedging.
   Beispiele:
   - 'orange und weiße Abstimmkarten' OK, wenn das Workshop-Setting die Funktion trägt
   - 'Boeing 777' OK, wenn der Schriftzug am Rumpf lesbar ist
   - 'runde orangefarbene Gegenstände' OK, wenn die Funktion wirklich nicht erkennbar ist
   - 'vermutlich Stimmkarten' / 'wirkt wie eine Dose' NICHT (Hedge statt Entscheidung)
   - 'Medikamentendose' NICHT, wenn nur eine Zylinderform ohne weiteren Beleg sichtbar ist

3. KEINE INTERAKTIONS-GESCHICHTEN: Wenn das Inventar nur 'Hund-Cartoon' + 'Laptop' listet,
   schreibe nicht 'Hund arbeitet am Laptop'. Du erfindest eine Handlung. Erlaubt: 'Hund-
   Cartoon, daneben ein Laptop.' Punkt.

4. IDENTIFIZIEREN WENN KLAR, NICHT RATEN WENN UNKLAR: Eine eindeutig erkennbare Spezies,
   Marke oder ein Modell wird benannt (klar lesbares Logo, eindeutige Lackierung,
   lesbarer Schriftzug). Auch ein UNVERWECHSELBARES PRODUKTDESIGN zählt als Beleg:
   Ein Produkt, das ein durchschnittlicher sehender Mensch am Design sofort erkennt
   (z.B. ein MacBook am charakteristischen flachen Aluminiumgehäuse), wird benannt —
   auch ohne lesbaren Schriftzug. Gegenprobe: ein generischer dunkler Laptop ohne
   solche Merkmale bleibt 'ein Laptop' und wird NICHT zum MacBook.
   Ist es UNKLAR ('stilisiertes Tier, Spezies unklar'), dann NICHT
   'Katze' oder 'Hund' raten, sondern 'Tier' bzw. die im Inventar gelistete
   Mehrfach-Hypothese.

5. FOTOMONTAGEN UND COLLAGEN: Wenn Bildelemente erkennbar nicht zusammenpassen
   (harte Freisteller-Kanten, widersprüchliche Schatten, Perspektiven oder Maßstäbe,
   Stilbruch zwischen Foto und Grafik, unmögliche Kombinationen wie ein berühmtes
   Bauwerk in fremder Landschaft), benenne das Bild ausdrücklich als Fotomontage
   oder Collage und beschreibe die Bestandteile getrennt. Eindeutig erkennbare
   eingefügte Motive werden benannt (Beispiel: 'Fotomontage: der Kölner Dom steht
   in einem Wüstencanyon'). Das gilt AUSDRÜCKLICH auch für fotorealistische
   Montagen ohne sichtbare Kanten oder Stilbruch: Die sachliche UNMÖGLICHKEIT
   der Kombination ist selbst der Indikator. Erkennst du ein Wahrzeichen oder
   Objekt an einem Ort, an dem es real nicht stehen kann, dann unterdrücke die
   Erkennung NICHT als Unsicherheit — benenne beides und kennzeichne das Bild
   als Fotomontage. Eine Montage als reales Foto zu beschreiben ist ein
   schwerer Fehler. Die Kennzeichnung erfolgt WOERTLICH mit dem Wort
   'Fotomontage' oder 'Collage' im Alt-Text (bewaehrter Auftakt:
   'Fotomontage: ...') — Umschreibungen wie 'aufgesetzte', 'eingefuegte'
   oder 'montierte' Elemente ersetzen die woertliche Kennzeichnung NICHT."""
