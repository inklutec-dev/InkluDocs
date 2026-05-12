"""Regeln für Lizenz- und Zertifizierungs-Logos.

E6-Korrektur (Steve-Review, 04.05.2026): 'ZÄHLE' ersetzt durch
'PRÜFE einzeln' — LLMs zählen unzuverlässig, Item-für-Item-Prüfung
vermeidet '3 Symbole = BY-SA'-Fehler.
"""

LIZENZ_LOGOS_REGELN = """LIZENZ- UND ZERTIFIZIERUNGS-LOGOS — KRITISCHE PRÄZISION:

Bei Logos die Lizenzen, Zertifizierungen oder Gütesiegel darstellen, MUSS der
exakte Lizenz- oder Zertifikatstyp benannt werden. Diese tragen rechtliche oder
qualitätsbezogene Information — Vereinfachung ist NICHT zulässig.

CREATIVE-COMMONS-LOGOS — Symbol für Symbol prüfen:
- CC = Creative Commons (Doppel-C im Kreis) — IMMER vorhanden
- BY = Attribution (Personen-Symbol)
- NC = NonCommercial (durchgestrichenes Dollarzeichen)
- SA = ShareAlike (Kreislauf-Pfeil)
- ND = NoDerivatives (Gleichheitszeichen)

REGEL: PRÜFE einzeln, welche der 5 möglichen CC-Symbole (CC, BY, NC, SA, ND)
sichtbar sind. Liste sie einzeln auf. Erst NACH dieser Auflistung den
Lizenz-Code zusammensetzen. LLMs zählen unzuverlässig — explizite
Item-für-Item-Prüfung vermeidet 'ich sehe 3 Symbole, also BY-SA'-Fehler.
- CC sichtbar? BY sichtbar? NC sichtbar? SA sichtbar? ND sichtbar?
  → Aus den ja-markierten Symbolen den Code zusammensetzen.
- Beispiel: CC=ja, BY=ja, NC=ja, ND=ja, SA=nein → 'Creative Commons BY-NC-ND'
- Beispiel: CC=ja, BY=ja, SA=ja, NC=nein, ND=nein → 'Creative Commons BY-SA'
- Wenn ein Symbol nicht klar lesbar → markieren und im Zweifel
  'Creative Commons Logo, Lizenztyp nicht eindeutig erkennbar'

ANDERE ZERTIFIZIERUNGS-LOGOS:
- Bio-Siegel: konkretes Siegel benennen (EU-Bio-Logo, Demeter, Bioland, Naturland etc.)
- Fair-Trade: konkrete Variante (Fairtrade International, GEPA, etc.)
- TÜV: konkrete Prüfung wenn lesbar (TÜV-geprüfte Sicherheit, GS-Zeichen etc.)
- Datenschutz: ePrivacyseal, TÜV-Datenschutz-Zertifikat etc.

NIEMALS verwechseln NC mit SA, ND mit SA, oder nicht-Lizenz-Logos als Lizenz-Logos
benennen.
"""
