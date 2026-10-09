"""Anzeigenamen der Chatbot-Werkzeuge (Pruefung 3 Barrierefreiheit, 30.09.2026, M1): die Live-Zeile „Ruft gerade auf: …“ und
„Genutzt: …“ unter jeder Antwort sagten rohe Namen wie „dokument_stand“ (VoiceOver liest die Unterstriche). Die Namen kommen
jetzt vom Server (window.WERKZEUG_NAMEN in app.html, uebersetzt in der Sprache der Oberflaeche). Jedes Werkzeug braucht hier
einen Namen — tests/test_chatbot_oberflaeche.py faellt sonst."""
from __future__ import annotations

from typing import Callable, Optional

# Werkzeugname -> deutscher Anzeigename (msgid, 6 Sprachen in backend/locales)
WERKZEUG_NAMEN: dict[str, str] = {
    # Bilder / Alt-Texte
    "list_project_images": "Bilderliste", "get_image_metadata": "Bild-Details", "view_image": "Bild ansehen",
    "generate_alt_text": "Alt-Text generieren", "update_alt_text": "Alt-Text speichern", "revert_alt_text": "Alt-Text zurücksetzen",
    "tavily_search": "Websuche",
    # Formularfelder / Quickinfos
    "list_form_fields": "Feldliste", "get_field_details": "Feld-Details", "view_field": "Feld ansehen",
    "generate_quickinfo": "Quickinfo generieren", "update_quickinfo": "Quickinfo speichern", "revert_quickinfo": "Quickinfo zurücksetzen",
    "search_master_data": "Stammdaten suchen", "save_to_master_data": "In Stammdaten übernehmen",
    # PDF
    "dokument_stand": "Dokumentstand", "barrierefrei_machen": "Barrierefrei machen",
    "komplett_barrierefrei_machen": "Komplett barrierefrei machen", "hoerprobe_lesen": "Hörprobe",
    "pruefung_starten": "KI-basierte Prüfung", "pruefbericht_lesen": "Bericht der KI-basierten Prüfung",
    "exportiere_fertige_pdf": "PDF herunterladen", "korrektur_anwenden": "Korrektur", "korrektur_rueckgaengig": "Korrektur rückgängig",
    "dokument_umbenennen": "Dokument umbenennen", "dokument_loeschen": "Dokument löschen", "alt_sprache_setzen": "Sprache der Alt-Texte",
    # Word, Ablage, Uebersetzen
    "pruefe_word_dokument": "Word-Prüfbericht", "konvertiere_zu_pdfua": "In barrierefreie PDF umwandeln",
    "exportiere_word": "Als Word herunterladen", "analysiere_word_struktur": "Struktur-Lektor",
    "uebersetze_dokument": "Übersetzen", "uebersetzung_stand": "Stand der Übersetzung",
    "exportiere_uebersetzung": "Übersetzung herunterladen", "liste_ausgaben": "Ablage", "lies_ausgabe": "Ablage-Eintrag lesen",
    # wie die Oberflaeche (30.09.2026)
    "testweise_taggen": "Testweise taggen", "pruefdatei_erstellen": "Prüfung starten",
    "pruefdatei_lesen": "Ergebnis der Barrierefreiheitsprüfung", "exportiere_alt_texte": "Alt-Texte herunterladen",
    "exportiere_quickinfos": "Quickinfos herunterladen", "alt_texte_generieren": "Alt-Texte generieren",
    "quickinfos_generieren": "Quickinfos generieren", "stammdaten_anwenden": "Stammdaten anwenden",
    "ki_kontext_setzen": "KI-Kontext", "eigener_prompt": "Gespeicherte Prompts", "ausgabe_loeschen": "Ablage-Eintrag löschen",
    # Grafik- und Webseiten-Projekte (InkluAgent-Ausbau Runde 1, Schritt 4)
    "bild_umbenennen": "Bild umbenennen", "bild_loeschen": "Bild löschen",
}


def werkzeug_name(name: str, _: Optional[Callable[[str], str]] = None) -> str:
    _ = _ or (lambda s: s)
    return _(WERKZEUG_NAMEN[name]) if name in WERKZEUG_NAMEN else _("Werkzeug")


def fuer_oberflaeche(_: Optional[Callable[[str], str]] = None) -> dict:
    """window.WERKZEUG_NAMEN: alle Namen in der Sprache der Oberflaeche."""
    _ = _ or (lambda s: s)
    return {k: _(v) for k, v in WERKZEUG_NAMEN.items()}
