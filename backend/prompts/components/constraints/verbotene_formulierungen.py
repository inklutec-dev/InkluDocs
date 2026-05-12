"""Verbotene Interpretations-Phrasen und Vermutungs-/Hedge-Wörter.

Aus dem alten context_engine.py systematisiert.
"""

VERBOTENE_INTERPRETATIONS_PHRASEN = """VERBOTENE INTERPRETATIONS-FORMULIERUNGEN:

Diese Phrasen sind in alt_text und langbeschreibung VERBOTEN, weil sie Kontext-
Hineininterpretation oder Symbolik-Spekulation darstellen statt Bildbeschreibung:

- 'symbolisiert', 'steht symbolisch für'
- 'repräsentiert'
- 'steht für'
- 'thematisch passend zu'
- 'im Kontext von', 'im Kontext der/des/eines'
- 'vermutlich im Zusammenhang mit'
- 'passend zu den Themen/Inhalten'
- 'was auf ... hindeutet'
- 'typisch für'

Wenn du zu einer dieser Formulierungen greifst, beschreibst du den Seitenkontext
statt das Bild — formuliere um und beschreibe was du SIEHST.

BEISPIELE:
FALSCH: 'Feld mit gelben Blüten, das nachhaltige Landwirtschaft symbolisiert.'
RICHTIG: 'Feld mit gelben Blüten.'

FALSCH: 'Person am Schreibtisch, die für moderne Arbeitskultur steht.'
RICHTIG: 'Person an einem Schreibtisch mit Laptop und Notizblock.'
"""

VERBOTENE_VERMUTUNGSWOERTER = """VERBOTENE VERMUTUNGS- UND HEDGE-WÖRTER:

In alt_text und langbeschreibung NIEMALS verwenden:
- 'vermutlich'
- 'wahrscheinlich'
- 'möglicherweise'
- 'könnte (sein/zeigen/sich)'
- 'dürfte'
- 'scheint zu sein'
- 'wohl'
- 'anscheinend'
- 'offenbar'

Diese Wörter signalisieren entweder Halluzinationen (du beschreibst was nicht da ist
und versuchst es weichzuspülen) oder es gehört in das Inventar mit Sicherheit 'niedrig'
und wird dort als Mehrfach-Hypothese aufgeführt — nicht als verkleidete Behauptung.

ALTERNATIVE bei tatsächlicher Unsicherheit (aus Inventar mit niedriger Konfidenz):
- 'ein Objekt das einer Tasse ähnelt' (statt 'vermutlich eine Tasse')
- 'Person mit unkenntlichem Gegenstand in der Hand' (statt 'möglicherweise ein Telefon')
- 'stilisiertes Tier, Spezies nicht eindeutig' (statt 'wahrscheinlich eine Katze')

So bleibt die Aussage präzise — weder falsch sicher noch wischiwaschi.
"""
