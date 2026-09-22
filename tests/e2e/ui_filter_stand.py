#!/usr/bin/env python3
"""Klicktest Filterleiste in jedem Projekt + ehrliche Laufmeldung + Platzhalter (17.09.2026).
Fassung 2 (12:45): Chips Alle / Offen / Alt-Text nach der Herunterladen-Regel (Michael Karbe).

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
    seite.goto(f"{B}/app?projekt={PID}&ansicht=alttexte", wait_until="networkidle")
    seite.wait_for_timeout(2500)
    check("Bild-Ansicht geladen, keine Skriptfehler", not probleme, probleme[:3])

    # Stand laut Server (dieselbe Quelle wie die Oberflaeche)
    bilder = seite.evaluate(f"fetch('/api/projects/{PID}').then(r => r.json()).then(d => d.images || [])")
    gesamt = len(bilder)
    n_error = sum(1 for b in bilder if b.get("status") == "error")
    # Spiegel von textstand()/_exportable_alt_text: sichtbarer Text = eigener Text vor KI-Text vor Quelltext
    def stand(b):
        t = b["alt_text_edited"] if b.get("alt_text_edited") is not None else (b.get("alt_text") or b.get("original_alt") or "")
        if t == "dekorativ" or (b.get("image_type") == "dekorativ" and not t.strip()): return "dekorativ"
        return "mit_text" if t.strip() else "offen"
    offen_ids = [f"imgcard_{b['id']}" for b in bilder if stand(b) == "offen"]
    mit_ids = [f"imgcard_{b['id']}" for b in bilder if stand(b) == "mit_text"]
    check("Projekt hat Bilder, das gestellte Fehlerbild und offene Bilder", gesamt > 0 and n_error >= 1 and offen_ids, (gesamt, n_error, len(offen_ids)))

    # 1) Filterleiste im Solo-Projekt
    check("Filterleiste ist da (Solo-Projekt)", seite.locator("#imageFilterBar").count() == 1)
    keys = seite.evaluate("Array.from(document.querySelectorAll('input[name=imgFilterStand]')).map(i => i.value)")
    check("Genau drei Grundchips: alle, offen, mit_text", keys == ["alle", "offen", "mit_text"], keys)
    legende = seite.locator("#filterStandFieldset legend").inner_text().strip()
    check("Legende heisst „Nach Bearbeitungsstand filtern“", "Bearbeitungsstand" in legende, legende)
    check("Solo-Projekt: kein Feld „Nach Freigabestatus filtern“", seite.locator("#filterPruefFieldset").count() == 0)
    check("Chip „Alle“ vorgewaehlt", seite.locator("input[name=imgFilterStand][value=alle]").is_checked())
    check("Zaehler Alle = Bilderzahl", chip_count(seite, "alle") == gesamt, (chip_count(seite, "alle"), gesamt))
    check("Zaehler Offen = Bilder ohne Text (Herunterladen-Regel)", chip_count(seite, "offen") == len(offen_ids), (chip_count(seite, "offen"), len(offen_ids)))
    check("Zaehler Alt-Text = Bilder mit Text", chip_count(seite, "mit_text") == len(mit_ids), (chip_count(seite, "mit_text"), len(mit_ids)))

    # 2) Filtern per Chip
    seite.check("input[name=imgFilterStand][value=offen]")
    seite.wait_for_timeout(300)
    sicht = sichtbare_karten(seite)
    check("Chip Offen zeigt genau die Bilder ohne Text", sorted(sicht) == sorted(offen_ids), (sicht, offen_ids))
    erg = seite.locator("#filterResult").inner_text()
    check("Ergebniszeile nennt n von m", re.search(r"\b%d\b.*\b%d\b" % (len(offen_ids), gesamt), erg) is not None, erg)
    check("Fehlerkarte traegt Abzeichen „Fehler“", seite.locator(f"#imgcard_{ERR_ID} .badge", has_text="Fehler").count() >= 1)

    seite.check("input[name=imgFilterStand][value=mit_text]")
    seite.wait_for_timeout(300)
    check("Chip Alt-Text zeigt genau die Bilder mit Text", sorted(sichtbare_karten(seite)) == sorted(mit_ids), (sichtbare_karten(seite), mit_ids))

    seite.check("input[name=imgFilterStand][value=alle]")
    seite.wait_for_timeout(300)
    check("Chip Alle zeigt wieder alle Karten", len(sichtbare_karten(seite)) == gesamt, len(sichtbare_karten(seite)))
    check("Fokus blieb auf dem Radio (kein Kontextwechsel)", seite.evaluate("document.activeElement && document.activeElement.name === 'imgFilterStand'"))

    # 3) Platzhalter immer bei leerem Feld
    ohne = seite.evaluate("Array.from(document.querySelectorAll('textarea.alt-text-field')).filter(t => !t.disabled && !(t.getAttribute('placeholder')||'').trim()).length")
    check("Jedes aktive Alt-Text-Feld hat einen Platzhalter", ohne == 0, ohne)
    ph = seite.evaluate(f"(document.getElementById('alttext_{ERR_ID}')||{{}}).getAttribute('placeholder')")
    check("Platzhalter-Text am Fehlerbild", ph and "Noch kein Alt-Text" in ph, ph)

    # 4) Ableitung des Textstands (reine Funktion, Spiegel der Herunterladen-Regel)
    stand_js = seite.evaluate("""() => [
        textstand({status:'error', alt_text:'', original_alt:'', alt_text_edited:null}),
        textstand({status:'error', alt_text:'', alt_text_edited:'Handtext'}),
        textstand({status:'pending', alt_text:'', original_alt:'', alt_text_edited:null}),
        textstand({status:'done', alt_text:'KI', original_alt:'', alt_text_edited:''}),
        textstand({status:'done', alt_text:'KI', alt_text_edited:null}),
        textstand({status:'done', alt_text:'', original_alt:'Quelle', alt_text_edited:null}),
        textstand({status:'done', alt_text:'', original_alt:'', alt_text_edited:null, image_type:'dekorativ'}),
        textstand({status:'done', alt_text:'dekorativ', alt_text_edited:null}),
        textstand({status:'done', alt_text:'   ', alt_text_edited:null}),
    ]""")
    check("textstand: fehler ohne/mit Handtext, pending, geleert, KI, Quelle, dekorativ ×2, nur Leerzeichen",
          stand_js == ["offen", "mit_text", "offen", "offen", "mit_text", "mit_text", "dekorativ", "dekorativ", "offen"], stand_js)

    # 5) Zaehler nach Einzel-Generieren (nur Client-Daten, kein Server): ein offenes Bild bekommt Text
    off_id = int(offen_ids[0].replace("imgcard_", ""))
    vorher = (chip_count(seite, "offen"), chip_count(seite, "mit_text"))
    seite.evaluate(f"updateFilterImageStatus({off_id}, 'done', 'Testtext')")
    nachher = (chip_count(seite, "offen"), chip_count(seite, "mit_text"))
    seite.evaluate(f"(() => {{ const i = filterImages.find(x => x.id === {off_id}); i.status = 'pending'; i.alt_text = ''; rebuildFilterChips(); }})()")
    zurueck = (chip_count(seite, "offen"), chip_count(seite, "mit_text"))
    check("Zaehler folgt Einzel-Generieren (offen→Alt-Text→offen)", vorher == zurueck and nachher == (vorher[0] - 1, vorher[1] + 1), (vorher, nachher, zurueck))
    check("Karte bleibt beim Zaehler-Update sichtbar", f"imgcard_{off_id}" in sichtbare_karten(seite))

    # 6) Geteiltes Projekt: Pruefchips folgen hinter den Grundchips (nur Client-Flag)
    keys_shared = seite.evaluate("() => { window._inReview = true; rebuildFilterChips(); const s = Array.from(document.querySelectorAll('input[name=imgFilterStand]')).map(i => i.value); const p = Array.from(document.querySelectorAll('input[name=imgFilterPruef]')).map(i => i.value); const leg = (document.querySelector('#filterPruefFieldset legend') || {}).textContent || ''; window._inReview = false; rebuildFilterChips(); const weg = !document.getElementById('filterPruefFieldset'); return { s, p, leg, weg }; }")
    check("Mit Freigabe: Feld 1 unveraendert, Feld 2 „Nach Freigabestatus filtern“ mit Alle + 6 Pruefchips", keys_shared["s"] == ["alle", "offen", "mit_text"] and keys_shared["p"] == ["alle", "neu", "in_bearbeitung", "lek_frei", "lek_aend", "her_frei", "her_aend"] and "Freigabestatus" in keys_shared["leg"] and keys_shared["weg"], keys_shared)

    # 7) Laufmeldung-Saetze und Uebersetzungen
    s1 = seite.evaluate("fehlgeschlagenSatz(1)")
    s2 = seite.evaluate("fehlgeschlagenSatz(2)")
    check("Laufmeldung Einzahl mit Hinweis auf Filter Offen", "Ein Bild" in s1 and "Offen" in s1, s1)
    check("Laufmeldung Mehrzahl mit Zahl", "2" in s2 and "Offen" in s2, s2)
    satz = seite.evaluate("t('{done} von {total} Alt-Texten generiert.', { done: 138, total: 140 })")
    check("Zaehlsatz wird gefuellt", "138" in satz and "140" in satz, satz)
    neue = ["Offen", "Alt-Text", "Nach Bearbeitungsstand filtern", "{done} von {total} Alt-Texten generiert.",
            "Ein Bild ist fehlgeschlagen.", "{n} Bilder sind fehlgeschlagen.", "Zu finden über den Filter „Offen“."]
    treffer = seite.evaluate("(keys) => keys.filter(k => k in (window.I18N || {})).length", neue)
    check("Sieben neue Texte liegen in window.I18N", treffer == 7, treffer)

    check("Keine Skriptfehler waehrend des Tests", not probleme, probleme[:3])
    br.close()

print(f"Ergebnis: {ok} OK, {fehler} FEHLER")
sys.exit(1 if fehler else 0)
