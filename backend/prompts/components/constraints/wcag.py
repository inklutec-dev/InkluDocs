"""WCAG 2.2 Grund- und Pflichtregeln für die Beschreibungs-Pässe."""

WCAG_GRUNDPRINZIP = """WCAG 2.2 Erfolgskriterium 1.1.1 (Nicht-Text-Inhalt, Stufe A):
Jeder bedeutungstragende Inhalt benötigt eine Text-Alternative die DEN GLEICHEN ZWECK
erfüllt wie das Bild für sehende Nutzer. Nicht 'beschreibt das Bild' — sondern 'erfüllt
den gleichen Zweck'. Ein Werbebild braucht eine andere Alt-Text-Strategie als ein
Dokumentations-Foto."""


WCAG_KONKRET_PFLICHTEN = """Konkrete Pflichten aus WCAG, BIK und EN 301 549:
- Lesbare Telefonnummern, Adressen, E-Mail, URLs MÜSSEN in den Alt-Text wenn sie im Bild sind
- Lizenz-Logos (Creative Commons, TÜV, Bio etc.) müssen mit korrektem Lizenz-Code benannt werden
- Bei Diagrammen: Spitzenwerte, Tiefstwerte, Trend müssen genannt sein
- Bei Tabellen: Spaltenköpfe + Endwerte müssen genannt sein
- Bei Personen-Fotos: Namen NUR aus Bild-Beschriftung oder explizitem Kontext, NIEMALS aus
  Gesichtserkennung (Persönlichkeitsrecht, DSGVO)
- Bei Karten: vollständig markierte Standorte, nicht nur Hintergrund-Geografie"""
