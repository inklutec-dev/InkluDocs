"""Anbieter-Profil der v4-Pipeline: alles, was am Aufruf oder am Prompt vom
Anbieter abhängt, an genau einer Stelle.

Eingeführt am 09.09.2026 mit der Umstellung des Erzeugers auf Gemini 3.1 Pro.
Die Grundprompts (prompts/) bleiben anbieterneutral: ein Text für alle. Was ein
Anbieter zusätzlich braucht, steht hier als Profil:

- Aufrufparameter, die der jeweilige Client auswertet (Temperatur, Reihenfolge
  von Bild und Text, Bildauflösung für das Modell).
- Ein kurzer Prompt-Zusatz, den der Orchestrator an den Combo-Prompt hängt
  (Feinschliff aus dem Prüfkorpus dieses Anbieters, keine neuen Grundregeln).

Ein neuer Anbieter bekommt ein neues Profil, kein Umbau der Pipeline. Für
Messungen lassen sich die Felder je Lauf per Umgebung übersteuern:

  V4_PROFIL_TEMPERATUR      z. B. 1.0 (leer = Profilwert)
  V4_PROFIL_BILD_ZUERST     on/off
  V4_PROFIL_BILDAUFLOESUNG  z. B. MEDIA_RESOLUTION_HIGH (Gemini), leer = Vorgabe
  V4_PROFIL_ZUSATZ          on/off (Prompt-Zusatz des Profils senden)
"""
from __future__ import annotations

import os
from dataclasses import dataclass, replace
from typing import Optional


@dataclass(frozen=True)
class AnbieterProfil:
    name: str
    # None = Temperatur des Aufrufers (Orchestrator) unverändert weiterreichen.
    temperatur: Optional[float]
    # Bild vor dem Text senden (Empfehlung von Google für Einzelbild-Prompts).
    bild_zuerst: bool
    # Gemini: generationConfig.mediaResolution; None = Vorgabe des Anbieters.
    bildaufloesung: Optional[str]
    # Zusatzblock am Ende des Combo-Prompts; '' = keiner.
    prompt_zusatz: str


# Feinschliff für Gemini aus dem Prüfkorpus (33 Bilder, 08.09.2026) und Michaels
# Befunden: Farben verallgemeinert, Material und Ort geraten, Deutung als Fakt,
# Kernfakten nur in der Langbeschreibung. Keine neuen Grundregeln, nur Betonung.
GEMINI_ZUSATZ = """BESONDERE SORGFALT

- Farben nennst du so, wie sie im Bild stehen: goldfarben statt gelb, türkis
  statt blau, dunkelblau statt blau-grau. Bei Datengrafiken nur dort, wo eine
  Farbe etwas erklärt.
- Material, Untergrund und Gewässerart nennst du nur, wenn das Bild sie zeigt:
  kein "hölzern" für einen grauen Kasten, kein "asphaltiert" für einen
  Schotterweg, kein "Seeufer", wenn es auch ein Fluss sein kann.
- Eine Deutung bleibt Beschreibung: rötliches Licht ist rötliches Licht, keine
  Dämmerung; ein Raum mit Flipcharts ist ein Raum mit Flipcharts, kein Seminar,
  solange der Kontext es nicht sagt.
- Orte, Bauwerke, Fachbegriffe (korinthische Säule, Balkendiagramm), Künstler
  und Jahr stehen im Alt-Text, nicht nur in der Langbeschreibung.
- Sind Objekte am Bildrand angeschnitten oder teils verdeckt, ist die Zahl
  keine exakte Zahl: "mindestens 23 Schalen, einige am Rand angeschnitten"
  (Belegregel 7)."""


PROFILE: dict[str, AnbieterProfil] = {
    'bedrock': AnbieterProfil('bedrock', temperatur=None, bild_zuerst=False, bildaufloesung=None, prompt_zusatz=''),
    'gemini': AnbieterProfil('gemini', temperatur=None, bild_zuerst=False, bildaufloesung=None, prompt_zusatz=GEMINI_ZUSATZ),
    'openai': AnbieterProfil('openai', temperatur=None, bild_zuerst=False, bildaufloesung=None, prompt_zusatz=''),
}


def _schalter(name: str) -> Optional[bool]:
    wert = os.environ.get(name, '').strip().lower()
    if wert in ('on', '1', 'true', 'ja'):
        return True
    if wert in ('off', '0', 'false', 'nein'):
        return False
    return None


def profil(anbieter: str) -> AnbieterProfil:
    """Profil des Anbieters, mit Übersteuerung per Umgebung (nur für Messungen)."""
    p = PROFILE.get((anbieter or '').strip().lower(), PROFILE['bedrock'])
    temp = os.environ.get('V4_PROFIL_TEMPERATUR', '').strip()
    if temp:
        try:
            p = replace(p, temperatur=float(temp))
        except ValueError:
            pass
    bild = _schalter('V4_PROFIL_BILD_ZUERST')
    if bild is not None:
        p = replace(p, bild_zuerst=bild)
    aufl = os.environ.get('V4_PROFIL_BILDAUFLOESUNG', '').strip()
    if aufl:
        p = replace(p, bildaufloesung=aufl)
    zusatz = _schalter('V4_PROFIL_ZUSATZ')
    if zusatz is False:
        p = replace(p, prompt_zusatz='')
    return p


def prompt_zusatz_block(anbieter: str) -> str:
    """Der Zusatzblock für den Combo-Prompt, mit Leerzeilen davor; '' wenn keiner."""
    z = profil(anbieter).prompt_zusatz.strip()
    return f'\n\n\n{z}' if z else ''
