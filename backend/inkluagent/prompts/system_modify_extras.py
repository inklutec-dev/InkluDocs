"""Bildtyp-spezifische Zusatzbloecke fuer den Modify-Modus.

Diese Bloecke werden NICHT in den Haupt-Prompt eingebaut, sondern vom
Code (chat_engine._modify_one_image) als zweite system-Nachricht an
Pixtral angehaengt, wenn die Bedingung passt.

So bleibt der Hauptprompt schlank, und Pixtral bekommt Spezialwissen
nur dort wo es wirklich gebraucht wird.

Stand 05.05.2026 — Hybrid-Loesung nach ChatGPT-Strategie-Empfehlung
(Runde 4): Schuesselchen-Symptom adressieren ohne Ueberoptimierung.
"""

# Behaelter-Wortliste fuer die Heuristik im Code (nicht in den Prompt).
CONTAINER_WORDS = [
    "schale", "schalen", "schuessel", "schuesselchen",
    "tasse", "tassen", "becher",
    "glas", "glaeser", "weinglas",
    "dose", "dosen",
    "box", "boxen",
    "teller", "tellerchen",
    "flasche", "flaschen",
    "kanne", "kannen",
    "vase", "vasen",
    "topf", "toepfe",
    "kelch", "krug", "humpen",
    "behaelter", "behaeltnis", "gefaess",
]

# Bildtypen die IMMER den Behaelter-Block bekommen (kommen aus Pipeline-Klassifikation).
CONTAINER_IMAGE_TYPES = {"foto_objekte"}


SYSTEM_MODIFY_BEHAELTER_BLOCK = """ZUSATZREGEL: BEI BEHAELTERN, GEFAESSEN ODER GLASIERTEN OBERFLAECHEN

Bei Behaeltern wie Schalen, Tassen, Glaesern, Dosen, Boxen, Tellern, Flaschen, Bechern, Vasen, Toepfen:

Inhalte, Fluessigkeiten, Fuellungen oder Substanzen NUR nennen, wenn sie im Bild eindeutig sichtbar oder im vorhandenen Alt-Text/Projektkontext eindeutig belegt sind.

Wenn nicht eindeutig, verwende:
"Innenraum"
"Innenflaeche"
"helle Oberflaeche"
"sichtbarer Innenbereich"

Verboten ohne eindeutigen Beleg:
"gefuellt"
"Fuellung"
"Fluessigkeit"
"Substanz"
"cremig"
"Creme"
"Pulver"
"Paste"
"Schaum"
"Masse"
"enthaelt"

Material nur nennen, wenn eindeutig sichtbar oder belegt. Bei Unsicherheit:
"helles, glattes Material"
"glaenzende Oberflaeche"
"glasierte Oberflaeche"
statt "Keramik", "Porzellan", "Metall", "Glas" zu raten."""
