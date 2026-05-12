"""Regeln für Eigennamen und Ortsnamen — Bild hat Vorrang vor Kontext."""

EIGENNAMEN_REGELN = """EIGENNAMEN UND ORTSNAMEN — Bild hat Vorrang vor Kontext:

Wenn ein Eigenname oder Ortsname im Bild lesbar ist und im Kontext anders steht,
hat der im Bild lesbare Text Vorrang. Häufige OCR-Verwechslungen:
- TURKU (finnische Stadt) ist NICHT Turkey (englisch für Türkei)
- Berlin vs. Berkeley (ähnlicher Anfang)
- Bonn vs. Bern (ähnlich kurz)

PRÜFE im Inventar: lesbare_texte hat Eigennamen mit Typ 'logo' oder 'beschriftung'.
Diese MÜSSEN wortgetreu übernommen werden — auch wenn der Kontext einen anderen
ähnlichen Namen nennt. Bei Mehrdeutigkeit dem Bild trauen, nicht dem Kontext.
"""
