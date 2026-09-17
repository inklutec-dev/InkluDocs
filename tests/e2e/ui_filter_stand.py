#!/usr/bin/env python3
"""Klicktest Filterleiste in jedem Projekt + ehrliche Laufmeldung + Platzhalter (17.09.2026).

Aufruf: /home/claude/.venv-pw/bin/python ui_filter_stand.py <projekt-id> <fehler-bild-id>
Vorher im Container: setup_ui_filter_stand.py <projekt-id> auf  (liefert die Fehler-Bild-ID),
nachher: ... ab. Erwartet ein Solo-Projekt (keine Freigabe) des E2E-Kontos mit Bildern.
Alles, was hier ausser dem Setup passiert, ist reine Oberflaeche — keine Serverdaten aendern sich."""
import os
import re
import sys
from playwright.sync_api import sync_playwright

B = os.environ.get("INKLUDOCS_E2E_URL", "https://staging.inkludocs.inklutec.de")
MAIL, PW = os.environ.get("INKLUDOCS_E2E_MAIL", ""), os.environ.get("INKLUDOCS_E2E_PW", "")
PID = sys.argv[1]
ERR_ID = int(sys.argv[2])
ok = fehler = 0


def check(name, cond, info=""):
    global ok, fehler
    if cond:
        ok += 1; print("OK   ", name)
    else:
        fehler += 1; print("FEHLT", name, "—", str(info)[:250])


def chip_count(seite, key):
    lab = seite.locator(f"label.filter-chip:has(input[value='{key}'])").first
    m = re.search(r"\((\d+)\)\s*$", lab.inner_text().strip())
    return int(m.group(1)) if m else -1


def sichtbare_karten(seite):
    return seite.evaluate("Array.from(document.querySelectorAll('section.image-review')).filter(s => !s.hidden && !s.closest('[hidden]')).map(s => s.id)")


with sync_playwright() as p:
    br = p.chromium.launch()
    seite = br.new_page()
    probleme = []
    seite.on("console", lambda m: probleme.append(m.text) if m.type == "error" else None)
    seite.on("pageerror", lambda e: probleme.append(str(e)))

    seite.goto(f"{B}/login", wait_until="domcontentloaded")
    seite.fill("input[type=email]", MAIL)
    seite.fill("input[type=password]", PW)
    seite.click("button[type=submit]")
    seite.wait_for_load_state("networkidle")
    seite.goto(f"{B}/app?projekt={PID}", wait_until="networkidle")
    seite.wait_for_timeout(2500)
    check("Bild-Ansicht geladen, keine Skriptfehler", not probleme, probleme[:3])

    # Stand laut Server (dieselbe Quelle wie die Oberflaeche)
    bilder = seite.evaluate(f"fetch('/api/projects/{PID}').then(r => r.json()).then(d => d.images || [])")
    gesamt = len(bilder)
    n_error = sum(1 for b in bilder if b.get("status") == "error")
    n_pending = sum(1 for b in bilder if b.get("status") == "pending")
    # fertig, aber ohne jeden Text und nie angefasst (alt_text_edited None) — zaehlt als „noch nicht generiert"
    n_leer = sum(1 for b in bilder if b.get("status") not in ("error", "pending", "processing")
                 and b.get("alt_text_edited") is None
                 and not (b.get("alt_text") or b.get("original_alt")))
    erwartet_nicht = n_pending + n_leer
    check("Projekt hat Bilder und das gestellte Fehlerbild", gesamt > 0 and n_error >= 1, (gesamt, n_error))

    # 1) Filterleiste im Solo-Projekt
    check("Filterleiste ist da (Solo-Projekt)", seite.locator("#imageFilterBar").count() == 1)
    keys = seite.evaluate("Array.from(document.querySelectorAll('input[name=imgFilterStatus]')).map(i => i.value)")
    check("Genau drei Grundchips: alle, nicht_generiert, fehlgeschlagen", keys == ["alle", "nicht_generiert", "fehlgeschlagen"], keys)
    legende = seite.locator("#filterStatusFieldset legend").inner_text().strip()
    check("Legende heisst „Nach Bearbeitungsstand filtern“", "Bearbeitungsstand" in legende, legende)
    check("Chip „Alle“ vorgewaehlt", seite.locator("input[name=imgFilterStatus][value=alle]").is_checked())
    check("Zaehler Alle = Bilderzahl", chip_count(seite, "alle") == gesamt, (chip_count(seite, "alle"), gesamt))
    check("Zaehler Fehlgeschlagen = Fehlerbilder", chip_count(seite, "fehlgeschlagen") == n_error, (chip_count(seite, "fehlgeschlagen"), n_error))
    check("Zaehler Noch nicht generiert = pending + textlos", chip_count(seite, "nicht_generiert") == erwartet_nicht, (chip_count(seite, "nicht_generiert"), erwartet_nicht))

    # 2) Filtern per Chip
    seite.check("input[name=imgFilterStatus][value=fehlgeschlagen]")
    seite.wait_for_timeout(300)
    sicht = sichtbare_karten(seite)
    check("Chip Fehlgeschlagen zeigt nur das Fehlerbild", sicht == [f"imgcard_{ERR_ID}"], sicht)
    erg = seite.locator("#filterResult").inner_text()
    check("Ergebniszeile nennt n von m", re.search(r"\b1\b.*\b%d\b" % gesamt, erg) is not None, erg)
    check("Fehlerkarte traegt Abzeichen „Fehler“", seite.locator(f"#imgcard_{ERR_ID} .badge", has_text="Fehler").count() >= 1)

    seite.check("input[name=imgFilterStatus][value=nicht_generiert]")
    seite.wait_for_timeout(300)
    check("Chip Noch nicht generiert zeigt erwartete Anzahl", len(sichtbare_karten(seite)) == erwartet_nicht, (len(sichtbare_karten(seite)), erwartet_nicht))

    seite.check("input[name=imgFilterStatus][value=alle]")
    seite.wait_for_timeout(300)
    check("Chip Alle zeigt wieder alle Karten", len(sichtbare_karten(seite)) == gesamt, len(sichtbare_karten(seite)))
    check("Fokus blieb auf dem Radio (kein Kontextwechsel)", seite.evaluate("document.activeElement && document.activeElement.name === 'imgFilterStatus'"))

    # 3) Platzhalter immer bei leerem Feld
    ohne = seite.evaluate("Array.from(document.querySelectorAll('textarea.alt-text-field')).filter(t => !t.disabled && !(t.getAttribute('placeholder')||'').trim()).length")
    check("Jedes aktive Alt-Text-Feld hat einen Platzhalter", ohne == 0, ohne)
    ph = seite.evaluate(f"(document.getElementById('alttext_{ERR_ID}')||{{}}).getAttribute('placeholder')")
    check("Platzhalter-Text am Fehlerbild", ph and "Noch kein Alt-Text" in ph, ph)

    # 4) Ableitung des Bearbeitungsstands (reine Funktion)
    stand = seite.evaluate("""() => [
        bearbeitungsstand({status:'error'}),
        bearbeitungsstand({status:'pending'}),
        bearbeitungsstand({status:'done', alt_text:'', original_alt:'', alt_text_edited:null}),
        bearbeitungsstand({status:'done', alt_text:'KI', original_alt:'', alt_text_edited:''}),
        bearbeitungsstand({status:'done', alt_text:'KI', alt_text_edited:null}),
        bearbeitungsstand({status:'processing', alt_text:'', alt_text_edited:null}),
    ]""")
    check("bearbeitungsstand: error/pending/textlos/geleert/fertig/laufend",
          stand == ["fehlgeschlagen", "nicht_generiert", "nicht_generiert", "generiert", "generiert", "generiert"], stand)

    # 5) Zaehler nach Einzel-Generieren (nur Client-Daten, kein Server)
    vorher = chip_count(seite, "fehlgeschlagen")
    seite.evaluate(f"updateFilterImageStatus({ERR_ID}, 'done', 'Testtext')")
    nachher = chip_count(seite, "fehlgeschlagen")
    seite.evaluate(f"updateFilterImageStatus({ERR_ID}, 'error')")
    zurueck = chip_count(seite, "fehlgeschlagen")
    check("Zaehler folgt Einzel-Generieren (error→done→error)", (vorher, nachher, zurueck) == (n_error, n_error - 1, n_error), (vorher, nachher, zurueck))
    check("Karte bleibt beim Zaehler-Update sichtbar", f"imgcard_{ERR_ID}" in sichtbare_karten(seite))

    # 6) Geteiltes Projekt: Pruefchips folgen hinter den Grundchips (nur Client-Flag)
    keys_shared = seite.evaluate("() => { window._inReview = true; rebuildFilterChips(); const k = Array.from(document.querySelectorAll('input[name=imgFilterStatus]')).map(i => i.value); window._inReview = false; rebuildFilterChips(); return k; }")
    check("Mit Freigabe: 3 Grundchips + 6 Pruefchips", keys_shared == ["alle", "nicht_generiert", "fehlgeschlagen", "neu", "in_bearbeitung", "lek_frei", "lek_aend", "her_frei", "her_aend"], keys_shared)

    # 7) Laufmeldung-Saetze und Uebersetzungen
    s1 = seite.evaluate("fehlgeschlagenSatz(1)")
    s2 = seite.evaluate("fehlgeschlagenSatz(2)")
    check("Laufmeldung Einzahl", "Ein Bild" in s1 and "Fehlgeschlagen" in s1, s1)
    check("Laufmeldung Mehrzahl mit Zahl", "2" in s2 and "Fehlgeschlagen" in s2, s2)
    satz = seite.evaluate("t('{done} von {total} Alt-Texten generiert.', { done: 138, total: 140 })")
    check("Zaehlsatz wird gefuellt", "138" in satz and "140" in satz, satz)
    neue = ["Noch nicht generiert", "Fehlgeschlagen", "Nach Bearbeitungsstand filtern", "{done} von {total} Alt-Texten generiert.",
            "Ein Bild ist fehlgeschlagen.", "{n} Bilder sind fehlgeschlagen.", "Zu finden über den Filter „Fehlgeschlagen“."]
    treffer = seite.evaluate("(keys) => keys.filter(k => k in (window.I18N || {})).length", neue)
    check("Sieben neue Texte liegen in window.I18N", treffer == 7, treffer)

    check("Keine Skriptfehler waehrend des Tests", not probleme, probleme[:3])
    br.close()

print(f"Ergebnis: {ok} OK, {fehler} FEHLER")
sys.exit(1 if fehler else 0)
