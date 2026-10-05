"""EXPRESS-SERVICE Stufe 1 (05.10.2026, Steve/Michael) — Kern ohne HTTP.

Kunden, die ihre Dokumente lieber von Profis barrierefrei aufbereiten oder pruefen lassen, legen PDFs aus ihren
Projekten (oder frisch hochgeladen) in einen Warenkorb, waehlen je Dokument die Leistung, machen ein paar Angaben,
bestaetigen Bedingungen und menschliche Bearbeitung und bestellen zahlungspflichtig in Credits. Geliefert wird
innerhalb einer Frist (Standard 48 Stunden; dem Kunden wird KEINE Uhrzeit genannt — Steve 05.10.2026).

Ablauf und Zustaende (Tabelle express_auftraege, siehe database.init_db):
    entwurf  -> der Warenkorb (hoechstens einer je Konto)
    neu      -> bestellt; Credits VORGEMERKT (billing.vorgemerkt), Originale in den Auftragsordner kopiert
    in_arbeit-> ein Bearbeiter hat uebernommen
    rueckfrage -> Frage an den Kunden; seine Antwort setzt zurueck auf in_arbeit (bzw. neu)
    geliefert -> Ergebnisse freigegeben; Credits in EINER Transaktion mit dem Statuswechsel ABGEBUCHT
    storniert -> Vormerkung faellt weg (nichts abgebucht); nach der Lieferung nicht mehr moeglich
Leistungen je Dokument: „aufbereiten“ (barrierefrei machen, Pruefung inklusive) und „pruefen“ (nur Pruefbericht).
Preise je Seite und Frist sind Einstellungen (system_kv 'express_einstellungen'); Standard sind PLATZHALTER.

Sicherheit: Jede Kundenfunktion bekommt die user_id des angemeldeten Kontos und prueft den Besitz in der SQL-Abfrage
selbst (kein „erst laden, dann vergleichen“). Dateipfade entstehen nur aus Zahlen (Auftrag, Position), nie aus
Dateinamen des Kunden. Hochgeladene Ergebnisse muessen echte, lesbare PDFs sein.

Doku: docs/EXPRESS_SERVICE.md. Endpunkte: express_api.py.
"""
from __future__ import annotations

import html
import json
import logging
import os
import re
import shutil
import sqlite3
import threading
from datetime import datetime, timedelta, timezone

import billing
import umsatz
from database import get_db

log = logging.getLogger("express")

# ─── Feste Werte ─────────────────────────────────────────────────────────

ENTWURF, NEU, IN_ARBEIT, RUECKFRAGE, GELIEFERT, STORNIERT = "entwurf", "neu", "in_arbeit", "rueckfrage", "geliefert", "storniert"
OFFEN = billing.EXPRESS_OFFEN                      # Credits vorgemerkt
STATUS_KUNDE = {NEU: "Eingegangen", IN_ARBEIT: "In Bearbeitung", RUECKFRAGE: "Rückfrage an dich",
                GELIEFERT: "Geliefert", STORNIERT: "Storniert"}
STATUS_VERWALTUNG = {NEU: "Neu", IN_ARBEIT: "In Arbeit", RUECKFRAGE: "Rückfrage", GELIEFERT: "Geliefert",
                     STORNIERT: "Storniert"}
LEISTUNGEN = {"aufbereiten": "Barrierefrei aufbereiten (mit Prüfung)", "pruefen": "Nur prüfen (Prüfbericht)"}
AKTION_JE_LEISTUNG = {"aufbereiten": "express_aufbereiten", "pruefen": "express_pruefen"}

EINSTELLUNGEN_STANDARD = {
    "preis_aufbereiten": 50,       # Credits je Seite — PLATZHALTER bis Michael/Steve den Preis festlegen
    "preis_pruefen": 25,           # Credits je Seite — PLATZHALTER
    "frist_stunden": 48,           # Lieferzusage; intern Grundlage fuer Erinnerung und „ueberfaellig“
    "max_seiten_auftrag": 500,     # groessere Auftraege nur auf Anfrage
    "max_dokumente_auftrag": 50,
    "team_mail": "",               # Benachrichtigungen an das Team; leer = NOTIFICATION_EMAIL des Servers
    "preise_festgelegt": False,    # False = Platzhalter; die Verwaltung zeigt dann einen Hinweis
}
_KV = "express_einstellungen"
ERINNERUNG_VORHER_STUNDEN = 12

# Fassung der beiden Pflicht-Haekchen. Bei jeder inhaltlichen Aenderung der Texte oder der Bedingungen hochzaehlen —
# im Auftrag steht, welcher Wortlaut galt (wie WIDERRUFSBELEHRUNG_FASSUNG in main.py).
ZUSTIMMUNG_FASSUNG = "2026-10-05-entwurf"
TEXT_BEDINGUNGEN = "Ich akzeptiere die Bedingungen für den Express-Service."
TEXT_BEARBEITUNG = ("Ich bin einverstanden, dass Mitarbeiter von InkluTec und unserem Partner Actino meine Dokumente "
                    "ansehen, bearbeiten und prüfen.")

MAX_ERGEBNIS_BYTES = 100 * 1024 * 1024
MAX_HINWEISE = 2000
MAX_TEXT = 2000
_TELEFON_RE = re.compile(r"^[0-9+()/\-. ]{3,40}$")
_IDEM_RE = re.compile(r"^[A-Za-z0-9\-]{8,64}$")
_MAIL_RE = re.compile(r"^[^\s@\x00-\x1f\x7f]+@[^\s@\x00-\x1f\x7f]+\.[^\s@\x00-\x1f\x7f]+$")

RESULTS_DIR = "/app/data/results"     # setzt express_api.build_router aus main.RESULTS_DIR
_bestell_sperre = threading.Lock()    # Pruefen-und-Vormerken atomar (ein Prozess); dazu BEGIN IMMEDIATE


class ExpressFehler(Exception):
    """Fachlicher Fehler mit verstaendlichem Text; status = HTTP-Status fuer den Router."""

    def __init__(self, text: str, status: int = 400, **extra):
        super().__init__(text)
        self.text, self.status, self.extra = text, status, extra


class NichtGefunden(ExpressFehler):
    def __init__(self, text: str = "Auftrag nicht gefunden"):
        super().__init__(text, 404)


# ─── Hilfen ──────────────────────────────────────────────────────────────

def _jetzt() -> datetime:
    return datetime.now(timezone.utc)


def _utc(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def _als_dt(text):
    if not text:
        return None
    try:
        return datetime.strptime(str(text)[:19], "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def _text(wert, name: str, maximal: int, pflicht: bool = False) -> str:
    t = str(wert or "").replace("\x00", "").strip()
    if pflicht and not t:
        raise ExpressFehler(f"Bitte {name} angeben.")
    if len(t) > maximal:
        raise ExpressFehler(f"{name} ist zu lang (höchstens {maximal} Zeichen).")
    return t


_ZEITFELDER = ("bestellt_am", "geliefert_am", "storniert_am", "faellig_am", "uebernommen_am", "zugestimmt_am", "am",
               "created_at", "ergebnis_am", "bericht_am")


def _lokal(d: dict) -> dict:
    """Zeitstempel (UTC in der Datenbank) fuer die Anzeige in deutsche Zeit 'JJJJ-MM-TT HH:MM' umrechnen — wie der
    Umsatz (umsatz.lokal). Nur fuer Ausgaben; gerechnet wird immer mit UTC."""
    for k in _ZEITFELDER:
        if d.get(k):
            d[k] = umsatz.lokal(d[k])
    return d


def ordner(user_id: int, auftrag_id: int) -> str:
    """Dateiordner eines Auftrags — nur aus Zahlen gebildet."""
    return os.path.join(RESULTS_DIR, str(int(user_id)), "_express", str(int(auftrag_id)))


def _datei_pfad(user_id: int, auftrag_id: int, pos_id: int, art: str) -> str:
    assert art in ("original", "ergebnis", "bericht")
    return os.path.join(ordner(user_id, auftrag_id), f"pos{int(pos_id)}_{art}.pdf")


def ist_pdf_datei(pfad: str) -> bool:
    try:
        with open(pfad, "rb") as f:
            return f.read(5) == b"%PDF-"
    except OSError:
        return False


def seitenzahl(pfad: str) -> int:
    """Seiten einer PDF; ExpressFehler, wenn sie nicht lesbar oder passwortgeschuetzt ist."""
    import fitz
    try:
        with fitz.open(pfad) as d:
            # Nur ein Oeffnungs-Passwort sperrt; reine Rechte-Verschluesselung (haeufig bei Behoerden-PDFs) ist kein Hindernis.
            if d.needs_pass:
                raise ExpressFehler("Die PDF ist mit einem Passwort geschützt und kann nicht bearbeitet werden.")
            return len(d)
    except ExpressFehler:
        raise
    except Exception:  # noqa: BLE001
        raise ExpressFehler("Die PDF konnte nicht gelesen werden.")


def _quelle_des_dokuments(doc: dict) -> str:
    """Die unveraenderte Kundendatei (roh_path), sonst die Arbeitsdatei."""
    for k in ("roh_path", "original_path"):
        p = (doc.get(k) or "").strip()
        if p and os.path.isfile(p):
            return p
    return ""


def _name_des_dokuments(doc: dict) -> str:
    return ((doc.get("display_name") or "").strip() or (doc.get("original_filename") or "").strip() or "Dokument")


# ─── Einstellungen ───────────────────────────────────────────────────────

def einstellungen() -> dict:
    e = dict(EINSTELLUNGEN_STANDARD)
    conn = get_db()
    try:
        r = conn.execute("SELECT value FROM system_kv WHERE key = ?", (_KV,)).fetchone()
    finally:
        conn.close()
    if r and r["value"]:
        try:
            gespeichert = json.loads(r["value"])
            if isinstance(gespeichert, dict):
                e.update({k: v for k, v in gespeichert.items() if k in EINSTELLUNGEN_STANDARD})
        except ValueError:
            log.warning("Express-Einstellungen unlesbar — Standard gilt")
    return e


def _ganzzahl(wert, name: str, minimum: int, maximum: int) -> int:
    text = str(wert if wert is not None else "").strip().replace(".", "").replace(" ", "")
    if not re.fullmatch(r"\d{1,7}", text):
        raise ExpressFehler(f"{name}: bitte eine ganze Zahl.")
    zahl = int(text)
    if not minimum <= zahl <= maximum:
        raise ExpressFehler(f"{name}: erlaubt sind {minimum} bis {maximum}.")
    return zahl


def speichere_einstellungen(daten: dict) -> dict:
    """Pruefen und speichern (nur Voll-Admins — die Rechte prueft der Router)."""
    e = einstellungen()
    e["preis_aufbereiten"] = _ganzzahl(daten.get("preis_aufbereiten"), "Preis „Barrierefrei aufbereiten“", 1, 100000)
    e["preis_pruefen"] = _ganzzahl(daten.get("preis_pruefen"), "Preis „Nur prüfen“", 1, 100000)
    e["frist_stunden"] = _ganzzahl(daten.get("frist_stunden"), "Frist", 1, 24 * 60)
    e["max_seiten_auftrag"] = _ganzzahl(daten.get("max_seiten_auftrag"), "Höchstzahl Seiten", 1, 100000)
    e["max_dokumente_auftrag"] = _ganzzahl(daten.get("max_dokumente_auftrag"), "Höchstzahl Dokumente", 1, 500)
    mail = str(daten.get("team_mail") or "").strip()
    if mail and (len(mail) > 254 or not _MAIL_RE.match(mail)):
        raise ExpressFehler("Benachrichtigung an: bitte eine gültige E-Mail-Adresse oder leer lassen.")
    e["team_mail"] = mail
    e["preise_festgelegt"] = daten.get("preise_festgelegt") is True
    conn = get_db()
    try:
        conn.execute("INSERT INTO system_kv (key, value, updated_at) VALUES (?, ?, datetime('now')) "
                     "ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at",
                     (_KV, json.dumps(e, ensure_ascii=False)))
        conn.commit()
    finally:
        conn.close()
    return e


def preis(leistung: str, seiten: int, e: dict = None) -> int:
    e = e or einstellungen()
    satz = e["preis_aufbereiten"] if leistung == "aufbereiten" else e["preis_pruefen"]
    return int(satz) * max(0, int(seiten or 0))


# ─── Warenkorb ───────────────────────────────────────────────────────────

def _warenkorb_id(conn, user_id: int, anlegen: bool):
    r = conn.execute("SELECT id FROM express_auftraege WHERE user_id = ? AND status = 'entwurf'", (user_id,)).fetchone()
    if r or not anlegen:
        return r["id"] if r else None
    conn.execute("INSERT OR IGNORE INTO express_auftraege (user_id, status) VALUES (?, 'entwurf')", (user_id,))
    conn.commit()
    return conn.execute("SELECT id FROM express_auftraege WHERE user_id = ? AND status = 'entwurf'", (user_id,)).fetchone()["id"]


def warenkorb(user_id: int) -> dict:
    """Der Warenkorb mit Positionen und Summen (Preise nach den aktuellen Einstellungen)."""
    e = einstellungen()
    conn = get_db()
    try:
        wid = _warenkorb_id(conn, user_id, anlegen=False)
        positionen = []
        if wid:
            # vorhanden = Dokument und Projekt gibt es noch (sonst kann nicht bestellt werden; der Kunde soll es sehen).
            positionen = [dict(r) for r in conn.execute(
                "SELECT x.id, x.project_id, x.document_id, x.dokument_name, x.seiten, x.leistung, "
                "(pr.id IS NOT NULL) AS vorhanden FROM express_positionen x "
                "LEFT JOIN documents d ON d.id = x.document_id "
                "LEFT JOIN projects pr ON pr.id = d.project_id AND pr.user_id = ? "
                "WHERE x.auftrag_id = ? ORDER BY x.id", (user_id, wid))]
    finally:
        conn.close()
    for p in positionen:
        p["credits"] = preis(p["leistung"], p["seiten"], e)
        p["vorhanden"] = bool(p["vorhanden"])
    return {"id": wid, "positionen": positionen, "dokumente": len(positionen),
            "seiten": sum(p["seiten"] for p in positionen), "credits": sum(p["credits"] for p in positionen)}


def _grenzen_pruefen(dokumente: int, seiten: int, e: dict) -> None:
    if dokumente > e["max_dokumente_auftrag"]:
        raise ExpressFehler(f"Ein Express-Auftrag kann höchstens {e['max_dokumente_auftrag']} Dokumente enthalten. "
                            "Für größere Aufträge schreib uns bitte.")
    if seiten > e["max_seiten_auftrag"]:
        raise ExpressFehler(f"Ein Express-Auftrag kann höchstens {e['max_seiten_auftrag']} Seiten enthalten. "
                            "Für größere Aufträge schreib uns bitte.")


def dokumente_hinzufuegen(user_id: int, document_ids: list) -> dict:
    """Dokumente des Kontos in den Warenkorb (Standard-Leistung „aufbereiten“). Doppelte werden uebersprungen.
    Rueckgabe {"hinzugefuegt": n, "uebersprungen": [Texte], "warenkorb": …}."""
    if not isinstance(document_ids, list) or not document_ids or len(document_ids) > 500:
        raise ExpressFehler("Bitte mindestens ein Dokument auswählen.")
    try:
        ids = sorted({int(x) for x in document_ids})
    except (TypeError, ValueError):
        raise ExpressFehler("Ungültige Auswahl.")
    e = einstellungen()
    conn = get_db()
    hinzu, hinweise = 0, []
    try:
        # Besitz im SQL: nur Dokumente aus Projekten DIESES Kontos.
        platz = ",".join("?" * len(ids))
        docs = {r["id"]: dict(r) for r in conn.execute(
            f"SELECT d.*, p.user_id AS besitzer FROM documents d JOIN projects p ON p.id = d.project_id "
            f"WHERE d.id IN ({platz}) AND p.user_id = ?", (*ids, user_id))}
        if len(docs) != len(ids):
            raise ExpressFehler("Mindestens ein Dokument wurde nicht gefunden.", 404)
        wid = _warenkorb_id(conn, user_id, anlegen=True)
        vorhanden = {r["document_id"] for r in conn.execute(
            "SELECT document_id FROM express_positionen WHERE auftrag_id = ?", (wid,))}
        stand = conn.execute("SELECT COUNT(*) AS n, COALESCE(SUM(seiten), 0) AS s FROM express_positionen WHERE auftrag_id = ?",
                             (wid,)).fetchone()
        n_dok, n_seiten = int(stand["n"]), int(stand["s"])
        neu = []
        for did in ids:
            doc = docs[did]
            name = _name_des_dokuments(doc)
            if did in vorhanden:
                hinweise.append(f"„{name}“ ist schon in deiner Auswahl.")
                continue
            quelle = _quelle_des_dokuments(doc)
            if not quelle or not ist_pdf_datei(quelle):
                hinweise.append(f"„{name}“ ist keine PDF und kann im Express-Service noch nicht bearbeitet werden.")
                continue
            try:
                seiten = seitenzahl(quelle)
            except ExpressFehler as fe:
                hinweise.append(f"„{name}“: {fe.text}")
                continue
            neu.append((did, doc["project_id"], name, seiten))
            n_dok += 1
            n_seiten += seiten
        _grenzen_pruefen(n_dok, n_seiten, e)
        for did, pid, name, seiten in neu:
            conn.execute("INSERT INTO express_positionen (auftrag_id, project_id, document_id, dokument_name, seiten, leistung) "
                         "VALUES (?, ?, ?, ?, ?, 'aufbereiten')", (wid, pid, did, name[:300], seiten))
            hinzu += 1
        conn.execute("UPDATE express_auftraege SET updated_at = datetime('now') WHERE id = ?", (wid,))
        conn.commit()
    finally:
        conn.close()
    return {"hinzugefuegt": hinzu, "hinweise": hinweise, "warenkorb": warenkorb(user_id)}


def leistung_setzen(user_id: int, pos_id: int, leistung: str) -> dict:
    if leistung not in LEISTUNGEN:
        raise ExpressFehler("Unbekannte Leistung.")
    conn = get_db()
    try:
        cur = conn.execute(
            "UPDATE express_positionen SET leistung = ? WHERE id = ? AND auftrag_id = "
            "(SELECT id FROM express_auftraege WHERE user_id = ? AND status = 'entwurf')", (leistung, int(pos_id), user_id))
        conn.commit()
    finally:
        conn.close()
    if cur.rowcount != 1:
        raise NichtGefunden("Dieses Dokument ist nicht in deiner Auswahl.")
    return warenkorb(user_id)


def position_entfernen(user_id: int, pos_id: int) -> dict:
    conn = get_db()
    try:
        cur = conn.execute(
            "DELETE FROM express_positionen WHERE id = ? AND auftrag_id = "
            "(SELECT id FROM express_auftraege WHERE user_id = ? AND status = 'entwurf')", (int(pos_id), user_id))
        conn.commit()
    finally:
        conn.close()
    if cur.rowcount != 1:
        raise NichtGefunden("Dieses Dokument ist nicht in deiner Auswahl.")
    return warenkorb(user_id)


def auto_projekt(user_id: int):
    """Projekt, in das der Express-Upload ohne Projekt legt (None = noch keins oder nicht mehr vorhanden)."""
    conn = get_db()
    try:
        r = conn.execute("SELECT a.auto_projekt_id FROM express_auftraege a JOIN projects p ON p.id = a.auto_projekt_id "
                         "AND p.user_id = a.user_id WHERE a.user_id = ? AND a.status = 'entwurf'", (user_id,)).fetchone()
        return int(r["auto_projekt_id"]) if r and r["auto_projekt_id"] else None
    finally:
        conn.close()


def auto_projekt_merken(user_id: int, project_id: int) -> int:
    """Das neu angelegte Projekt am Warenkorb merken und benennen („Express-Auftrag <Nr>“). Rueckgabe: Warenkorb-id."""
    conn = get_db()
    try:
        wid = _warenkorb_id(conn, user_id, anlegen=True)
        conn.execute("UPDATE express_auftraege SET auto_projekt_id = ? WHERE id = ? AND user_id = ?", (project_id, wid, user_id))
        conn.execute("UPDATE projects SET name = ? WHERE id = ? AND user_id = ?", (f"Express-Auftrag {wid}", project_id, user_id))
        conn.commit()
        return wid
    finally:
        conn.close()


# ─── Bestellen ───────────────────────────────────────────────────────────

class KeinGuthaben(ExpressFehler):
    def __init__(self, preis_credits: int, verfuegbar: int):
        super().__init__(f"Für diesen Auftrag brauchst du {preis_credits} Credits, verfügbar sind {verfuegbar}. "
                         "Du kannst Credits als Paket dazukaufen.", 402, preis=preis_credits, verfuegbar=verfuegbar)


def bestellen(user_id: int, *, ansprechpartner, telefon, hinweise, bedingungen, bearbeitung, idempotenz,
              sprache: str = "de", texte: dict = None, absender: str = "") -> dict:
    """Warenkorb zahlungspflichtig bestellen. Rueckgabe {"auftrag_id", "neu": bool} (neu=False: dieselbe Bestellung
    kam schon einmal an — Doppelklick, Netzwiederholung — und wird nicht doppelt angelegt)."""
    name = _text(ansprechpartner, "einen Ansprechpartner", 120, pflicht=True)
    tel = _text(telefon, "Telefon", 40)
    if tel and not _TELEFON_RE.match(tel):
        raise ExpressFehler("Telefon: bitte nur Ziffern, Leerzeichen und + ( ) / - verwenden.")
    notiz = _text(hinweise, "Hinweise", MAX_HINWEISE)
    if bedingungen is not True or bearbeitung is not True:
        raise ExpressFehler("Bitte beide Häkchen setzen: Bedingungen und Einverständnis zur Bearbeitung durch Menschen.")
    idem = str(idempotenz or "")
    if not _IDEM_RE.match(idem):
        raise ExpressFehler("Ungültige Anfrage, bitte die Seite neu laden.")
    texte = texte or {}
    e = einstellungen()
    with _bestell_sperre:
        conn = get_db()
        try:
            schon = conn.execute("SELECT id FROM express_auftraege WHERE user_id = ? AND idempotenz = ?",
                                 (user_id, idem)).fetchone()
            if schon:
                return {"auftrag_id": int(schon["id"]), "neu": False}
            wid = _warenkorb_id(conn, user_id, anlegen=False)
            # Quelle nur, solange Dokument UND Projekt noch diesem Konto gehoeren.
            positionen = [dict(r) for r in conn.execute(
                "SELECT p.*, CASE WHEN pr.id IS NULL THEN '' ELSE d.roh_path END AS roh_path, "
                "CASE WHEN pr.id IS NULL THEN '' ELSE d.original_path END AS original_path FROM express_positionen p "
                "LEFT JOIN documents d ON d.id = p.document_id "
                "LEFT JOIN projects pr ON pr.id = d.project_id AND pr.user_id = ? "
                "WHERE p.auftrag_id = ? ORDER BY p.id", (user_id, wid))] if wid else []
        finally:
            conn.close()
        if not positionen:
            raise ExpressFehler("Deine Auswahl ist leer. Bitte zuerst Dokumente hinzufügen.")
        # Seiten frisch zaehlen (die Datei kann sich seit dem Hinzufuegen geaendert haben) und Preis festschreiben.
        for p in positionen:
            quelle = _quelle_des_dokuments(p)
            if not quelle or not ist_pdf_datei(quelle):
                raise ExpressFehler(f"Das Dokument „{p['dokument_name']}“ gibt es nicht mehr. Bitte aus der Auswahl entfernen.")
            p["quelle"] = quelle
            p["seiten"] = seitenzahl(quelle)
            p["credits"] = preis(p["leistung"], p["seiten"], e)
        seiten = sum(p["seiten"] for p in positionen)
        summe = sum(p["credits"] for p in positionen)
        _grenzen_pruefen(len(positionen), seiten, e)
        konto = billing._konto_fuer(user_id)
        verfuegbar = billing.verfuegbare_credits(user_id)   # None = unbegrenzt; zieht andere Vormerkungen schon ab
        if verfuegbar is not None and verfuegbar < summe:
            raise KeinGuthaben(summe, verfuegbar)
        # Originale in den Auftragsordner kopieren: was die Profis bekommen, aendert sich nicht mehr, auch wenn der
        # Kunde das Projekt weiter bearbeitet oder loescht.
        ziel = ordner(user_id, wid)
        os.makedirs(ziel, exist_ok=True)
        try:
            for p in positionen:
                p["original_pfad"] = _datei_pfad(user_id, wid, p["id"], "original")
                shutil.copyfile(p["quelle"], p["original_pfad"])
        except OSError as fe:
            log.exception("Express: Original nicht kopiert (Auftrag %s)", wid)
            raise ExpressFehler("Die Dokumente konnten nicht übernommen werden. Bitte später erneut versuchen.", 500) from fe
        jetzt = _jetzt()
        faellig = jetzt + timedelta(hours=int(e["frist_stunden"]))
        conn = get_db()
        try:
            conn.isolation_level = None
            conn.execute("BEGIN IMMEDIATE")
            cur = conn.execute(
                "UPDATE express_auftraege SET status = 'neu', konto_user_id = ?, ansprechpartner = ?, telefon = ?, "
                "hinweise = ?, seiten_gesamt = ?, credits_gesamt = ?, frist_stunden = ?, faellig_am = ?, "
                "zustimmung_fassung = ?, zustimmung_bedingungen = ?, zustimmung_bearbeitung = ?, zustimmung_sprache = ?, "
                "zustimmung_absender = ?, zugestimmt_am = ?, idempotenz = ?, bestellt_am = ?, updated_at = ? "
                "WHERE id = ? AND user_id = ? AND status = 'entwurf'",
                (konto, name, tel, notiz, seiten, summe, int(e["frist_stunden"]), _utc(faellig), ZUSTIMMUNG_FASSUNG,
                 str(texte.get("bedingungen") or TEXT_BEDINGUNGEN)[:500], str(texte.get("bearbeitung") or TEXT_BEARBEITUNG)[:500],
                 (sprache or "de")[:10], (absender or "")[:80], _utc(jetzt), idem, _utc(jetzt), _utc(jetzt), wid, user_id))
            if cur.rowcount != 1:
                conn.execute("ROLLBACK")
                raise ExpressFehler("Die Auswahl hat sich gerade geändert. Bitte die Seite neu laden.", 409)
            for p in positionen:
                conn.execute("UPDATE express_positionen SET seiten = ?, credits = ?, original_pfad = ? WHERE id = ?",
                             (p["seiten"], p["credits"], p["original_pfad"], p["id"]))
            conn.execute("INSERT INTO express_verlauf (auftrag_id, art, text, von_name, fuer_kunde) VALUES (?, 'bestellt', ?, ?, 1)",
                         (wid, f"{len(positionen)} Dokumente, {seiten} Seiten, {summe} Credits vorgemerkt", name))
            conn.execute("COMMIT")
        except ExpressFehler:
            raise
        except Exception:
            try:
                conn.execute("ROLLBACK")
            except Exception:  # noqa: BLE001
                pass
            raise
        finally:
            conn.close()
    return {"auftrag_id": int(wid), "neu": True}


# ─── Lesen ───────────────────────────────────────────────────────────────

def _auftrag_zeile(conn, auftrag_id: int, user_id: int = None):
    sql = ("SELECT a.*, u.email AS kunde_email, COALESCE(NULLIF(TRIM(u.display_name), ''), u.email) AS kunde_name, "
           "k.email AS konto_email, COALESCE(NULLIF(TRIM(k.display_name), ''), k.email) AS konto_name "
           "FROM express_auftraege a LEFT JOIN users u ON u.id = a.user_id LEFT JOIN users k ON k.id = a.konto_user_id "
           "WHERE a.id = ? AND a.status != 'entwurf'")
    werte = [int(auftrag_id)]
    if user_id is not None:
        sql += " AND a.user_id = ?"
        werte.append(int(user_id))
    return conn.execute(sql, werte).fetchone()


def _faellig_info(a: dict) -> dict:
    """Fuer die Verwaltung: Stunden bis zur Frist (negativ = ueberfaellig). Nur fuer offene Auftraege."""
    if a["status"] not in OFFEN:
        return {"faellig_in_stunden": None, "ueberfaellig": False}
    f = _als_dt(a.get("faellig_am"))
    if not f:
        return {"faellig_in_stunden": None, "ueberfaellig": False}
    stunden = (f - _jetzt()).total_seconds() / 3600
    return {"faellig_in_stunden": round(stunden, 1), "ueberfaellig": stunden < 0}


def _position_dict(p: dict, fuer_kunde: bool, status: str) -> dict:
    try:
        vera = json.loads(p.get("verapdf") or "{}") or None
    except ValueError:
        vera = None
    d = {"id": p["id"], "project_id": p["project_id"], "document_id": p["document_id"],
         "dokument_name": p["dokument_name"], "seiten": p["seiten"], "leistung": p["leistung"],
         "leistung_text": LEISTUNGEN.get(p["leistung"], p["leistung"]), "credits": p["credits"]}
    if fuer_kunde:
        geliefert = status == GELIEFERT
        d["ergebnis_da"] = geliefert and bool(p["ergebnis_pfad"]) and os.path.isfile(p["ergebnis_pfad"])
        d["bericht_da"] = geliefert and bool(p["bericht_pfad"]) and os.path.isfile(p["bericht_pfad"])
        d["verapdf"] = ({"bestanden": bool(vera.get("bestanden")), "zusammenfassung": vera.get("zusammenfassung", "")}
                        if geliefert and vera and "bestanden" in vera else None)
    else:
        d.update({"original_da": bool(p["original_pfad"]) and os.path.isfile(p["original_pfad"]),
                  "ergebnis_da": bool(p["ergebnis_pfad"]) and os.path.isfile(p["ergebnis_pfad"]),
                  "ergebnis_name": p["ergebnis_name"], "ergebnis_am": p["ergebnis_am"], "ergebnis_von": p["ergebnis_von"],
                  "bericht_da": bool(p["bericht_pfad"]) and os.path.isfile(p["bericht_pfad"]),
                  "bericht_name": p["bericht_name"], "bericht_am": p["bericht_am"], "verapdf": vera})
    return d


def _verlauf(conn, auftrag_id: int, fuer_kunde: bool) -> list:
    sql = "SELECT art, text, von_name, fuer_kunde, created_at FROM express_verlauf WHERE auftrag_id = ?"
    if fuer_kunde:
        sql += " AND fuer_kunde = 1"
    eintraege = [dict(r) for r in conn.execute(sql + " ORDER BY created_at, id", (int(auftrag_id),))]
    if fuer_kunde:
        for v in eintraege:
            # Kunden sehen keine Namen der Bearbeiter, nur „InkluDocs“ bzw. sich selbst.
            v["von_name"] = "" if v["art"] in ("uebernommen", "rueckfrage", "geliefert", "storniert") else v["von_name"]
            v.pop("fuer_kunde", None)
    return eintraege


def auftrag_fuer_kunde(user_id: int, auftrag_id: int) -> dict:
    conn = get_db()
    try:
        a = _auftrag_zeile(conn, auftrag_id, user_id)
        if not a:
            raise NichtGefunden()
        a = dict(a)
        pos = [dict(r) for r in conn.execute("SELECT * FROM express_positionen WHERE auftrag_id = ? ORDER BY id", (a["id"],))]
        verlauf = _verlauf(conn, a["id"], True)
    finally:
        conn.close()
    _lokal(a)
    for v in verlauf:
        _lokal(v)
    return {
        "id": a["id"], "status": a["status"], "status_text": STATUS_KUNDE.get(a["status"], a["status"]),
        "bestellt_am": a["bestellt_am"], "geliefert_am": a["geliefert_am"], "storniert_am": a["storniert_am"],
        "storno_grund": a["storno_grund"] if a["status"] == STORNIERT else "",
        "frist_stunden": a["frist_stunden"], "ansprechpartner": a["ansprechpartner"], "telefon": a["telefon"],
        "hinweise": a["hinweise"], "seiten": a["seiten_gesamt"], "credits": a["credits_gesamt"],
        "credits_stand": ("abgebucht" if a["status"] == GELIEFERT else ("frei" if a["status"] == STORNIERT else "vorgemerkt")),
        "konto_name": a["konto_name"] if a["konto_user_id"] and a["konto_user_id"] != a["user_id"] else "",
        "zustimmung": {"bedingungen": a["zustimmung_bedingungen"], "bearbeitung": a["zustimmung_bearbeitung"],
                       "fassung": a["zustimmung_fassung"], "am": a["zugestimmt_am"]},
        "positionen": [_lokal(_position_dict(p, True, a["status"])) for p in pos],
        "verlauf": verlauf,
        "rueckfrage_offen": a["status"] == RUECKFRAGE,
    }


def auftraege_des_kunden(user_id: int) -> list:
    conn = get_db()
    try:
        rows = [dict(r) for r in conn.execute(
            "SELECT a.id, a.status, a.bestellt_am, a.geliefert_am, a.seiten_gesamt, a.credits_gesamt, a.frist_stunden, "
            "(SELECT COUNT(*) FROM express_positionen p WHERE p.auftrag_id = a.id) AS dokumente "
            "FROM express_auftraege a WHERE a.user_id = ? AND a.status != 'entwurf' ORDER BY a.id DESC", (user_id,))]
    finally:
        conn.close()
    for r in rows:
        r["status_text"] = STATUS_KUNDE.get(r["status"], r["status"])
        _lokal(r)
    return rows


def auftrag_fuer_verwaltung(auftrag_id: int) -> dict:
    conn = get_db()
    try:
        a = _auftrag_zeile(conn, auftrag_id)
        if not a:
            raise NichtGefunden()
        a = dict(a)
        pos = [dict(r) for r in conn.execute("SELECT * FROM express_positionen WHERE auftrag_id = ? ORDER BY id", (a["id"],))]
        verlauf = _verlauf(conn, a["id"], False)
    finally:
        conn.close()
    out = {k: a[k] for k in ("id", "user_id", "konto_user_id", "status", "ansprechpartner", "telefon", "hinweise",
                             "seiten_gesamt", "credits_gesamt", "frist_stunden", "faellig_am", "bestellt_am",
                             "bearbeiter_id", "bearbeiter_name", "uebernommen_am", "geliefert_am", "geliefert_von",
                             "storniert_am", "storniert_von", "storno_grund", "interne_notiz", "kunde_email", "kunde_name",
                             "konto_name", "zustimmung_fassung", "zustimmung_bedingungen", "zustimmung_bearbeitung",
                             "zustimmung_sprache", "zugestimmt_am")}
    out["status_text"] = STATUS_VERWALTUNG.get(a["status"], a["status"])
    out.update(_faellig_info(a))
    _lokal(out)
    out["positionen"] = [_lokal(_position_dict(p, False, a["status"])) for p in pos]
    out["verlauf"] = [_lokal(v) for v in verlauf]
    return out


def liste_fuer_verwaltung() -> dict:
    """Alle bestellten Auftraege nach Dringlichkeit: ueberfaellig, neu, in Arbeit, Rueckfrage, dann die letzten 100
    gelieferten und stornierten."""
    conn = get_db()
    try:
        offen = [dict(r) for r in conn.execute(
            "SELECT a.id, a.status, a.bestellt_am, a.faellig_am, a.seiten_gesamt, a.credits_gesamt, a.bearbeiter_name, "
            "COALESCE(NULLIF(TRIM(u.display_name), ''), u.email) AS kunde_name, "
            "(SELECT COUNT(*) FROM express_positionen p WHERE p.auftrag_id = a.id) AS dokumente "
            "FROM express_auftraege a LEFT JOIN users u ON u.id = a.user_id "
            "WHERE a.status IN ('neu', 'in_arbeit', 'rueckfrage') ORDER BY a.faellig_am, a.id")]
        fertig = [dict(r) for r in conn.execute(
            "SELECT a.id, a.status, a.bestellt_am, a.geliefert_am, a.storniert_am, a.seiten_gesamt, a.credits_gesamt, "
            "a.bearbeiter_name, COALESCE(NULLIF(TRIM(u.display_name), ''), u.email) AS kunde_name, "
            "(SELECT COUNT(*) FROM express_positionen p WHERE p.auftrag_id = a.id) AS dokumente "
            "FROM express_auftraege a LEFT JOIN users u ON u.id = a.user_id "
            "WHERE a.status IN ('geliefert', 'storniert') ORDER BY a.id DESC LIMIT 100")]
    finally:
        conn.close()
    gruppen = {"ueberfaellig": [], NEU: [], IN_ARBEIT: [], RUECKFRAGE: [], GELIEFERT: [], STORNIERT: []}
    for a in offen:
        a.update(_faellig_info(a))
        a["status_text"] = STATUS_VERWALTUNG[a["status"]]
        gruppen["ueberfaellig" if a["ueberfaellig"] and a["status"] != RUECKFRAGE else a["status"]].append(_lokal(a))
    for a in fertig:
        a["status_text"] = STATUS_VERWALTUNG[a["status"]]
        gruppen[a["status"]].append(_lokal(a))
    return gruppen


# ─── Verwaltung: Arbeitsschritte ─────────────────────────────────────────

def _verlauf_eintrag(conn, auftrag_id: int, art: str, text: str, von: str, fuer_kunde: bool) -> None:
    conn.execute("INSERT INTO express_verlauf (auftrag_id, art, text, von_name, fuer_kunde) VALUES (?, ?, ?, ?, ?)",
                 (auftrag_id, art, (text or "")[:MAX_TEXT], (von or "")[:120], 1 if fuer_kunde else 0))


def _zustand_wechseln(auftrag_id: int, erlaubt: tuple, setzen: str, werte: tuple, verlauf: tuple = None) -> dict:
    """UPDATE nur aus erlaubten Zustaenden (Vergleichen-und-Tauschen); NichtGefunden bzw. 409 sonst."""
    conn = get_db()
    try:
        platz = ",".join("?" * len(erlaubt))
        cur = conn.execute(f"UPDATE express_auftraege SET {setzen}, updated_at = datetime('now') "
                           f"WHERE id = ? AND status IN ({platz})", (*werte, int(auftrag_id), *erlaubt))
        if cur.rowcount != 1:
            gibt_es = conn.execute("SELECT status FROM express_auftraege WHERE id = ? AND status != 'entwurf'",
                                   (int(auftrag_id),)).fetchone()
            if not gibt_es:
                raise NichtGefunden()
            raise ExpressFehler(f"Das geht im Stand „{STATUS_VERWALTUNG.get(gibt_es['status'], gibt_es['status'])}“ nicht.", 409)
        if verlauf:
            _verlauf_eintrag(conn, int(auftrag_id), *verlauf)
        conn.commit()
    finally:
        conn.close()
    return auftrag_fuer_verwaltung(auftrag_id)


def uebernehmen(auftrag_id: int, person: dict) -> dict:
    """Bearbeiter setzen (auch Umhaengen auf eine andere Person); aus „neu“ wird „in Arbeit“."""
    return _zustand_wechseln(
        auftrag_id, OFFEN,
        "bearbeiter_id = ?, bearbeiter_name = ?, uebernommen_am = datetime('now'), "
        "status = CASE WHEN status = 'neu' THEN 'in_arbeit' ELSE status END",
        (person["id"], person["name"][:120]),
        ("uebernommen", "In Bearbeitung genommen", person["name"], True))


def rueckfrage(auftrag_id: int, person: dict, text) -> dict:
    """Frage an den Kunden. Die Frist ruht ab jetzt (rueckfrage_seit) und verlaengert sich bei der Antwort um die
    Wartezeit — so steht es in den Bedingungen."""
    frage = _text(text, "eine Frage", MAX_TEXT, pflicht=True)
    return _zustand_wechseln(auftrag_id, OFFEN, "status = 'rueckfrage', rueckfrage_seit = COALESCE(rueckfrage_seit, datetime('now'))",
                             (), ("rueckfrage", frage, person["name"], True))


def antwort_kunde(user_id: int, auftrag_id: int, text) -> dict:
    """Antwort des Kunden auf eine Rueckfrage: zurueck auf „in Arbeit“ (bzw. „neu“, wenn noch niemand uebernommen hat)."""
    antwort = _text(text, "deine Antwort", MAX_TEXT, pflicht=True)
    conn = get_db()
    try:
        # Frist um die Wartezeit verlaengern; Erinnerung/Ueberfaellig-Meldung gelten fuer die neue Frist neu.
        cur = conn.execute(
            "UPDATE express_auftraege SET status = CASE WHEN bearbeiter_id IS NULL AND bearbeiter_name = '' THEN 'neu' "
            "ELSE 'in_arbeit' END, "
            "faellig_am = CASE WHEN rueckfrage_seit IS NULL OR faellig_am IS NULL THEN faellig_am ELSE "
            "  datetime(faellig_am, '+' || CAST(MAX(0, (julianday('now') - julianday(rueckfrage_seit)) * 86400) AS INTEGER) || ' seconds') END, "
            "rueckfrage_seit = NULL, erinnert_am = NULL, ueberfaellig_gemeldet_am = NULL, updated_at = datetime('now') "
            "WHERE id = ? AND user_id = ? AND status = 'rueckfrage'",
            (int(auftrag_id), int(user_id)))
        if cur.rowcount != 1:
            if not _auftrag_zeile(conn, auftrag_id, user_id):
                raise NichtGefunden()
            raise ExpressFehler("Zu diesem Auftrag ist gerade keine Rückfrage offen.", 409)
        name = conn.execute("SELECT ansprechpartner FROM express_auftraege WHERE id = ?", (int(auftrag_id),)).fetchone()[0]
        _verlauf_eintrag(conn, int(auftrag_id), "antwort", antwort, name, True)
        conn.commit()
    finally:
        conn.close()
    return auftrag_fuer_kunde(user_id, auftrag_id)


def notiz_setzen(auftrag_id: int, person: dict, text) -> dict:
    notiz = _text(text, "die Notiz", MAX_TEXT)
    erlaubt = (NEU, IN_ARBEIT, RUECKFRAGE, GELIEFERT, STORNIERT)
    return _zustand_wechseln(auftrag_id, erlaubt, "interne_notiz = ?", (notiz,),
                             ("notiz", "Interne Notiz geändert", person["name"], False))


def datei_speichern(auftrag_id: int, pos_id: int, art: str, inhalt: bytes, dateiname: str, person: dict) -> dict:
    """Ergebnis (art='ergebnis') oder Pruefbericht (art='bericht') einer Position ablegen. Nur PDF, nur in offenen
    Auftraegen. Der Dateiname des Bearbeiters wird nur angezeigt, nie als Pfad benutzt."""
    if art not in ("ergebnis", "bericht"):
        raise ExpressFehler("Unbekannte Dateiart.")
    if not inhalt:
        raise ExpressFehler("Die Datei ist leer.")
    if len(inhalt) > MAX_ERGEBNIS_BYTES:
        raise ExpressFehler(f"Die Datei ist zu groß (höchstens {MAX_ERGEBNIS_BYTES // (1024 * 1024)} MB).", 413)
    if not inhalt.startswith(b"%PDF-"):
        raise ExpressFehler("Bitte eine PDF-Datei hochladen.")
    conn = get_db()
    try:
        r = conn.execute("SELECT a.user_id, a.status, p.id, p.dokument_name FROM express_positionen p "
                         "JOIN express_auftraege a ON a.id = p.auftrag_id WHERE p.id = ? AND a.id = ? AND a.status != 'entwurf'",
                         (int(pos_id), int(auftrag_id))).fetchone()
    finally:
        conn.close()
    if not r:
        raise NichtGefunden("Dokument im Auftrag nicht gefunden")
    if r["status"] not in OFFEN:
        raise ExpressFehler("Der Auftrag ist schon abgeschlossen.", 409)
    pfad = _datei_pfad(r["user_id"], auftrag_id, pos_id, art)
    os.makedirs(os.path.dirname(pfad), exist_ok=True)
    tmp = pfad + ".tmp"
    with open(tmp, "wb") as f:
        f.write(inhalt)
    try:
        seitenzahl(tmp)               # wirft bei unlesbarer oder verschluesselter PDF
    except ExpressFehler:
        os.remove(tmp)
        raise
    os.replace(tmp, pfad)
    anzeige = re.sub(r"[\x00-\x1f\x7f/\\]", "", os.path.basename(str(dateiname or ""))).strip()[:200] or f"{art}.pdf"
    conn = get_db()
    try:
        if art == "ergebnis":
            conn.execute("UPDATE express_positionen SET ergebnis_pfad = ?, ergebnis_name = ?, ergebnis_am = datetime('now'), "
                         "ergebnis_von = ?, verapdf = '' WHERE id = ?", (pfad, anzeige, person["name"][:120], int(pos_id)))
        else:
            conn.execute("UPDATE express_positionen SET bericht_pfad = ?, bericht_name = ?, bericht_am = datetime('now') "
                         "WHERE id = ?", (pfad, anzeige, int(pos_id)))
        _verlauf_eintrag(conn, int(auftrag_id), art, f"{'Ergebnis' if art == 'ergebnis' else 'Prüfbericht'} für "
                         f"„{r['dokument_name']}“ hochgeladen", person["name"], False)
        conn.execute("UPDATE express_auftraege SET updated_at = datetime('now') WHERE id = ?", (int(auftrag_id),))
        conn.commit()
    finally:
        conn.close()
    return {"pfad": pfad, "auftrag": auftrag_fuer_verwaltung(auftrag_id)}


def verapdf_merken(pos_id: int, ergebnis) -> None:
    """Ergebnis der automatischen veraPDF-Pruefung eines hochgeladenen Ergebnisses (None = nicht pruefbar)."""
    wert = {"nicht_geprueft": True} if not ergebnis else {
        "bestanden": bool(ergebnis.get("bestanden")), "zusammenfassung": str(ergebnis.get("zusammenfassung") or "")[:1000],
        "regeln_fehlgeschlagen": int(ergebnis.get("regeln_fehlgeschlagen") or 0)}
    conn = get_db()
    try:
        conn.execute("UPDATE express_positionen SET verapdf = ? WHERE id = ?", (json.dumps(wert, ensure_ascii=False), int(pos_id)))
        conn.commit()
    finally:
        conn.close()


def lieferbar(a: dict) -> dict:
    """Was vor dem Liefern fehlt oder auffaellt: {"fehlt": [Texte], "befunde": [Texte]}."""
    fehlt, befunde = [], []
    for p in a["positionen"]:
        if p["leistung"] == "aufbereiten" and not p["ergebnis_da"]:
            fehlt.append(f"„{p['dokument_name']}“: das aufbereitete Dokument fehlt noch.")
        if p["leistung"] == "pruefen" and not p["bericht_da"]:
            fehlt.append(f"„{p['dokument_name']}“: der Prüfbericht fehlt noch.")
        v = p.get("verapdf") or {}
        if p["ergebnis_da"] and v.get("bestanden") is False:
            befunde.append(f"„{p['dokument_name']}“: veraPDF meldet Abweichungen ({v.get('zusammenfassung') or 'siehe Bericht'}).")
    return {"fehlt": fehlt, "befunde": befunde}


def liefern(auftrag_id: int, person: dict, trotz_befunden: bool = False) -> dict:
    """Freigeben und abbuchen — in EINER Transaktion: Status „geliefert“ (nur aus offenen Zustaenden), je Leistung ein
    Verbrauchs-Ereignis auf den Topf der Bestellung, Paket-Abbuchung des Ueberhangs. Ein zweiter Klick findet den
    Auftrag nicht mehr offen und bucht nichts."""
    a = auftrag_fuer_verwaltung(auftrag_id)
    if a["status"] not in OFFEN:
        raise ExpressFehler("Der Auftrag ist schon abgeschlossen.", 409)
    pruef = lieferbar(a)
    if pruef["fehlt"]:
        raise ExpressFehler("Vor dem Liefern fehlt noch etwas: " + " ".join(pruef["fehlt"]), 400, fehlt=pruef["fehlt"])
    if pruef["befunde"] and not trotz_befunden:
        raise ExpressFehler("veraPDF meldet Abweichungen. Trotzdem liefern?", 409, befunde=pruef["befunde"], nachfrage=True)
    user_id = int(a["user_id"])
    konto = int(a["konto_user_id"] or user_id)
    posten = {}
    for p in a["positionen"]:
        aktion = AKTION_JE_LEISTUNG[p["leistung"]]
        posten[aktion] = posten.get(aktion, 0) + int(p["credits"] or 0)
    conn = get_db()
    try:
        conn.isolation_level = None
        conn.execute("BEGIN IMMEDIATE")
        cur = conn.execute("UPDATE express_auftraege SET status = 'geliefert', geliefert_am = datetime('now'), geliefert_von = ?, "
                           "updated_at = datetime('now') WHERE id = ? AND status IN ('neu', 'in_arbeit', 'rueckfrage')",
                           (person["name"][:120], int(auftrag_id)))
        if cur.rowcount != 1:
            conn.execute("ROLLBACK")
            raise ExpressFehler("Der Auftrag ist schon abgeschlossen.", 409)
        for aktion, credits in posten.items():
            if credits > 0:
                conn.execute("INSERT INTO usage_events (user_id, konto_user_id, quelle, aktion, credits, image_id) "
                             "VALUES (?, ?, 'express', ?, ?, NULL)", (user_id, konto, aktion, credits))
        billing._pakete_abbuchen(conn, konto)
        _verlauf_eintrag(conn, int(auftrag_id), "geliefert", "Ergebnisse stehen zum Herunterladen bereit", person["name"], True)
        conn.execute("COMMIT")
    except ExpressFehler:
        raise
    except Exception:
        try:
            conn.execute("ROLLBACK")
        except Exception:  # noqa: BLE001
            pass
        raise
    finally:
        conn.close()
    return auftrag_fuer_verwaltung(auftrag_id)


def stornieren(auftrag_id: int, person: dict, grund) -> dict:
    """Storno nur vor der Lieferung; die Vormerkung faellt mit dem Status weg (nichts wurde abgebucht)."""
    text = _text(grund, "einen Grund", 500, pflicht=True)
    return _zustand_wechseln(auftrag_id, OFFEN,
                             "status = 'storniert', storniert_am = datetime('now'), storniert_von = ?, storno_grund = ?",
                             (person["name"][:120], text), ("storniert", text, person["name"], True))


def datei_fuer_kunde(user_id: int, auftrag_id: int, pos_id: int, art: str):
    """(pfad, anzeigename) — nur eigene, gelieferte Auftraege."""
    if art not in ("ergebnis", "bericht"):
        raise NichtGefunden("Datei nicht gefunden")
    conn = get_db()
    try:
        r = conn.execute(f"SELECT p.{art}_pfad AS pfad, p.dokument_name FROM express_positionen p "
                         "JOIN express_auftraege a ON a.id = p.auftrag_id "
                         "WHERE p.id = ? AND a.id = ? AND a.user_id = ? AND a.status = 'geliefert'",
                         (int(pos_id), int(auftrag_id), int(user_id))).fetchone()
    finally:
        conn.close()
    if not r or not r["pfad"] or not os.path.isfile(r["pfad"]):
        raise NichtGefunden("Datei nicht gefunden")
    return r["pfad"], _download_name(r["dokument_name"], art)


def datei_fuer_verwaltung(auftrag_id: int, pos_id: int, art: str):
    if art not in ("original", "ergebnis", "bericht"):
        raise NichtGefunden("Datei nicht gefunden")
    conn = get_db()
    try:
        r = conn.execute(f"SELECT p.{art}_pfad AS pfad, p.dokument_name FROM express_positionen p "
                         "JOIN express_auftraege a ON a.id = p.auftrag_id WHERE p.id = ? AND a.id = ? AND a.status != 'entwurf'",
                         (int(pos_id), int(auftrag_id))).fetchone()
    finally:
        conn.close()
    if not r or not r["pfad"] or not os.path.isfile(r["pfad"]):
        raise NichtGefunden("Datei nicht gefunden")
    return r["pfad"], _download_name(r["dokument_name"], art)


def originale_fuer_zip(auftrag_id: int) -> list:
    """[(pfad, name_im_zip)] aller vorhandenen Originale; Namen eindeutig und ohne Pfadteile."""
    a = auftrag_fuer_verwaltung(auftrag_id)
    conn = get_db()
    try:
        rows = [dict(r) for r in conn.execute("SELECT id, original_pfad, dokument_name FROM express_positionen "
                                              "WHERE auftrag_id = ? ORDER BY id", (a["id"],))]
    finally:
        conn.close()
    out, namen = [], set()
    for i, r in enumerate(rows, 1):
        if r["original_pfad"] and os.path.isfile(r["original_pfad"]):
            name = f"{i:02d}_" + _download_name(r["dokument_name"], "original")
            while name in namen:
                name = f"{i:02d}_{len(namen)}_" + _download_name(r["dokument_name"], "original")
            namen.add(name)
            out.append((r["original_pfad"], name))
    return out


def _download_name(dokument_name: str, art: str) -> str:
    stamm = re.sub(r"\.pdf$", "", str(dokument_name or "Dokument"), flags=re.IGNORECASE)
    stamm = re.sub(r"[\x00-\x1f\x7f/\\:*?\"<>|]", "_", stamm).strip()[:150] or "Dokument"
    zusatz = {"original": "", "ergebnis": " (barrierefrei)", "bericht": " (Prüfbericht)"}[art]
    return f"{stamm}{zusatz}.pdf"


# ─── Bearbeiter-Recht ────────────────────────────────────────────────────

def bearbeiter_liste() -> list:
    conn = get_db()
    try:
        return [dict(r) for r in conn.execute("SELECT id, email, display_name FROM users WHERE express_bearbeiter = 1 "
                                              "ORDER BY display_name, email")]
    finally:
        conn.close()


def bearbeiter_setzen(email: str = None, user_id: int = None, an: bool = True) -> dict:
    conn = get_db()
    try:
        if email is not None:
            r = conn.execute("SELECT id, email, display_name FROM users WHERE lower(email) = lower(?)",
                             (str(email).strip(),)).fetchone()
        else:
            r = conn.execute("SELECT id, email, display_name FROM users WHERE id = ?", (int(user_id),)).fetchone()
        if not r:
            raise ExpressFehler("Zu dieser E-Mail-Adresse gibt es kein Konto. Bitte zuerst ein Konto anlegen.", 404)
        conn.execute("UPDATE users SET express_bearbeiter = ? WHERE id = ?", (1 if an else 0, r["id"]))
        conn.commit()
        return dict(r)
    finally:
        conn.close()


def ist_bearbeiter(user_id: int) -> bool:
    conn = get_db()
    try:
        r = conn.execute("SELECT express_bearbeiter FROM users WHERE id = ? AND is_active = 1", (int(user_id),)).fetchone()
        return bool(r and r["express_bearbeiter"])
    except sqlite3.OperationalError:
        return False
    finally:
        conn.close()


# ─── Erinnerungen ────────────────────────────────────────────────────────

def faellige_meldungen() -> list:
    """Auftraege, fuer die JETZT eine Team-Meldung faellig ist: [(art, auftrag_id)] mit art 'erinnerung' (12 Stunden vor
    der Frist) oder 'ueberfaellig'. Jede Meldung wird atomar beansprucht (Spalte gesetzt), damit sie nie doppelt geht —
    auch nicht bei mehreren Prozessen. Waehrend einer Rueckfrage wartet die Erinnerung (der Ball liegt beim Kunden)."""
    jetzt = _jetzt()
    bald = _utc(jetzt + timedelta(hours=ERINNERUNG_VORHER_STUNDEN))
    out = []
    conn = get_db()
    try:
        for r in conn.execute("SELECT id, faellig_am, erinnert_am, ueberfaellig_gemeldet_am FROM express_auftraege "
                              "WHERE status IN ('neu', 'in_arbeit') AND faellig_am IS NOT NULL AND faellig_am <= ?", (bald,)).fetchall():
            if r["faellig_am"] <= _utc(jetzt):
                if r["ueberfaellig_gemeldet_am"] is None:
                    cur = conn.execute("UPDATE express_auftraege SET ueberfaellig_gemeldet_am = datetime('now'), "
                                       "erinnert_am = COALESCE(erinnert_am, datetime('now')) "
                                       "WHERE id = ? AND ueberfaellig_gemeldet_am IS NULL", (r["id"],))
                    if cur.rowcount == 1:
                        out.append(("ueberfaellig", r["id"]))
            elif r["erinnert_am"] is None:
                cur = conn.execute("UPDATE express_auftraege SET erinnert_am = datetime('now') WHERE id = ? AND erinnert_am IS NULL",
                                   (r["id"],))
                if cur.rowcount == 1:
                    out.append(("erinnerung", r["id"]))
        conn.commit()
    finally:
        conn.close()
    return out


# ─── E-Mails (Texte; versandt wird ueber main.send_email) ─────────────────

def _mail_html(absaetze: list, link: str = "", link_text: str = "") -> str:
    teile = [f"<p>{html.escape(a)}</p>" for a in absaetze if a]
    if link:
        teile.append(f'<p><a href="{html.escape(link, quote=True)}">{html.escape(link_text or link)}</a></p>')
    teile.append("<p>Viele Grüße<br>InkluDocs</p>")
    return "\n".join(teile)


def mail_kunde(art: str, a: dict, basis_url: str, text: str = "") -> tuple:
    """(Betreff, HTML) fuer den Kunden — ohne Anhang, mit Link zur Auftragsuebersicht."""
    link = f"{basis_url.rstrip('/')}/express/auftrag/{a['id']}"
    hallo = f"Hallo {a.get('ansprechpartner') or a.get('kunde_name') or ''},".replace(" ,", ",")
    n = len(a.get("positionen") or [])
    if art == "bestellt":
        return (f"Dein Express-Auftrag {a['id']} ist eingegangen", _mail_html([
            hallo, f"wir haben deinen Auftrag erhalten: {n} Dokument{'e' if n != 1 else ''}, {a.get('seiten_gesamt') or a.get('seiten')} Seiten.",
            f"Dafür sind {a.get('credits_gesamt') or a.get('credits')} Credits vorgemerkt. Abgebucht wird erst bei der Lieferung.",
            f"Wir liefern innerhalb von {a.get('frist_stunden')} Stunden. Den Stand siehst du jederzeit in deiner Auftragsübersicht."],
            link, "Auftragsübersicht öffnen"))
    if art == "rueckfrage":
        return (f"Rückfrage zu deinem Express-Auftrag {a['id']}", _mail_html([
            hallo, "zu deinem Express-Auftrag haben wir eine Frage:", text,
            "Bitte antworte in deiner Auftragsübersicht. Dort steht die Frage auch."], link, "Zur Rückfrage"))
    if art == "geliefert":
        return (f"Dein Express-Auftrag {a['id']} ist fertig", _mail_html([
            hallo, "deine Dokumente sind fertig. Du kannst sie in deiner Auftragsübersicht herunterladen.",
            f"Abgebucht wurden {a.get('credits_gesamt') or a.get('credits')} Credits."], link, "Dokumente herunterladen"))
    if art == "storniert":
        return (f"Dein Express-Auftrag {a['id']} wurde storniert", _mail_html([
            hallo, "dein Express-Auftrag wurde storniert.", f"Grund: {text}" if text else "",
            "Die vorgemerkten Credits sind wieder frei. Abgebucht wurde nichts."], link, "Auftragsübersicht öffnen"))
    raise ValueError(art)


def mail_team(art: str, a: dict, basis_url: str, text: str = "") -> tuple:
    link = f"{basis_url.rstrip('/')}/verwaltung/express/{a['id']}"
    faellig = (a.get("faellig_am") or "")[:16]
    kopf = (f"Kunde: {a.get('kunde_name') or ''} ({a.get('kunde_email') or ''}), {len(a.get('positionen') or [])} Dokumente, "
            f"{a.get('seiten_gesamt')} Seiten, fällig am {faellig} (deutsche Zeit).")
    if art == "bestellt":
        leist = "; ".join(f"{p['dokument_name']}: {LEISTUNGEN[p['leistung']]}, {p['seiten']} Seiten" for p in a.get("positionen") or [])
        return (f"Neuer Express-Auftrag {a['id']}", _mail_html([kopf, f"Dokumente: {leist}",
                                                               f"Hinweise des Kunden: {a.get('hinweise')}" if a.get("hinweise") else ""],
                                                              link, "Auftrag in der Verwaltung öffnen"))
    if art == "antwort":
        return (f"Antwort auf die Rückfrage zu Express-Auftrag {a['id']}", _mail_html([kopf, f"Antwort: {text}"], link,
                                                                                     "Auftrag in der Verwaltung öffnen"))
    if art == "erinnerung":
        return (f"Erinnerung: Express-Auftrag {a['id']} ist in {ERINNERUNG_VORHER_STUNDEN} Stunden fällig",
                _mail_html([kopf, f"Bearbeiter: {a.get('bearbeiter_name') or 'noch niemand'}"], link, "Auftrag öffnen"))
    if art == "ueberfaellig":
        return (f"Überfällig: Express-Auftrag {a['id']}", _mail_html([kopf, "Die Lieferfrist ist abgelaufen.",
                                                                       f"Bearbeiter: {a.get('bearbeiter_name') or 'noch niemand'}"],
                                                                      link, "Auftrag öffnen"))
    raise ValueError(art)


def team_empfaenger(standard: str) -> list:
    """Benachrichtigungsadresse aus den Einstellungen (sonst die des Servers) plus alle Express-Bearbeiter; ohne Dubletten."""
    e = einstellungen()
    adressen = [e.get("team_mail") or standard] + [b["email"] for b in bearbeiter_liste()]
    out = []
    for x in adressen:
        x = (x or "").strip()
        if x and x.lower() not in {y.lower() for y in out}:
            out.append(x)
    return out


# ─── Nachweis (Auftragsuebersicht als barrierefreie PDF) ──────────────────

def _x(text) -> str:
    """Text fuer WordprocessingML (escapen, Steuerzeichen raus)."""
    return html.escape(re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", str(text or "")), quote=False)


def _absatz(text: str, stil: str = "", liste: bool = False) -> str:
    ppr = ""
    if stil or liste:
        ppr = "<w:pPr>" + (f'<w:pStyle w:val="{stil}"/>' if stil else "") + \
              ('<w:numPr><w:ilvl w:val="0"/><w:numId w:val="1"/></w:numPr>' if liste else "") + "</w:pPr>"
    return f'<w:p>{ppr}<w:r><w:t xml:space="preserve">{_x(text)}</w:t></w:r></w:p>'


_DOCX_STYLES = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
<w:docDefaults><w:rPrDefault><w:rPr><w:rFonts w:ascii="Liberation Sans" w:hAnsi="Liberation Sans" w:cs="Liberation Sans"/>
<w:sz w:val="22"/><w:lang w:val="de-DE"/></w:rPr></w:rPrDefault><w:pPrDefault><w:pPr><w:spacing w:after="120"/></w:pPr></w:pPrDefault></w:docDefaults>
<w:style w:type="paragraph" w:default="1" w:styleId="Normal"><w:name w:val="Normal"/></w:style>
<w:style w:type="paragraph" w:styleId="Heading1"><w:name w:val="heading 1"/><w:basedOn w:val="Normal"/><w:next w:val="Normal"/>
<w:pPr><w:keepNext/><w:spacing w:before="240" w:after="120"/><w:outlineLvl w:val="0"/></w:pPr><w:rPr><w:b/><w:sz w:val="36"/></w:rPr></w:style>
<w:style w:type="paragraph" w:styleId="Heading2"><w:name w:val="heading 2"/><w:basedOn w:val="Normal"/><w:next w:val="Normal"/>
<w:pPr><w:keepNext/><w:spacing w:before="240" w:after="80"/><w:outlineLvl w:val="1"/></w:pPr><w:rPr><w:b/><w:sz w:val="28"/></w:rPr></w:style>
<w:style w:type="paragraph" w:styleId="ListParagraph"><w:name w:val="List Paragraph"/><w:basedOn w:val="Normal"/><w:pPr><w:ind w:left="720"/></w:pPr></w:style>
</w:styles>"""

_DOCX_NUMBERING = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:numbering xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
<w:abstractNum w:abstractNumId="0"><w:lvl w:ilvl="0"><w:start w:val="1"/><w:numFmt w:val="bullet"/><w:lvlText w:val="•"/>
<w:lvlJc w:val="left"/><w:pPr><w:ind w:left="720" w:hanging="360"/></w:pPr></w:lvl></w:abstractNum>
<w:num w:numId="1"><w:abstractNumId w:val="0"/></w:num>
</w:numbering>"""


def nachweis_docx(a: dict, ziel: str, zeit_text) -> str:
    """Auftragsuebersicht als Word-Datei mit echten Ueberschriften (Heading 1/2) und Aufzaehlungen (numbering.xml);
    LibreOffice macht daraus die PDF/UA (pdfua_export.konvertiere, mit veraPDF-Pruefung). Ohne python-docx (nicht im
    Image): minimales OOXML von Hand. zeit_text: Zeitstempel -> Anzeigetext. Ausdruecklich KEINE Rechnung."""
    import zipfile
    titel = f"Express-Auftrag {a['id']} – Auftragsübersicht"
    teile = [_absatz(titel, "Heading1"),
             _absatz("Nachweis über einen Auftrag an den Express-Service von InkluDocs. Dies ist keine Rechnung: "
                     "Bezahlt wird mit Credits, deren Kauf gesondert abgerechnet wurde."),
             _absatz("Auftrag", "Heading2")]
    stand = {"vorgemerkt": "vorgemerkt", "abgebucht": "abgebucht", "frei": "wieder frei (storniert)"}[a["credits_stand"]]
    for zeile in (f"Auftragsnummer: {a['id']}", f"Bestellt am: {zeit_text(a['bestellt_am'])}", f"Stand: {a['status_text']}",
                  (f"Geliefert am: {zeit_text(a['geliefert_am'])}" if a.get("geliefert_am")
                   else f"Lieferung: innerhalb von {a['frist_stunden']} Stunden"),
                  f"Credits: {a['credits']} ({stand})", f"Seiten: {a['seiten']}"):
        teile.append(_absatz(zeile))
    if a.get("storno_grund"):
        teile.append(_absatz(f"Grund des Stornos: {a['storno_grund']}"))
    teile.append(_absatz("Angaben", "Heading2"))
    teile.append(_absatz(f"Ansprechpartner: {a['ansprechpartner']}"))
    if a.get("telefon"):
        teile.append(_absatz(f"Telefon: {a['telefon']}"))
    if a.get("hinweise"):
        teile.append(_absatz(f"Hinweise: {a['hinweise']}"))
    teile.append(_absatz("Dokumente", "Heading2"))
    for p in a["positionen"]:
        teile.append(_absatz(f"{p['dokument_name']}: {p['seiten']} Seiten, {p['leistung_text']}, {p['credits']} Credits",
                             "ListParagraph", liste=True))
    teile.append(_absatz("Einverständnis", "Heading2"))
    z = a["zustimmung"]
    teile.append(_absatz(f"Bestätigt am {zeit_text(z['am'])} (Fassung {z['fassung']}):"))
    teile.append(_absatz(z["bedingungen"], "ListParagraph", liste=True))
    teile.append(_absatz(z["bearbeitung"], "ListParagraph", liste=True))
    teile.append(_absatz("Verlauf", "Heading2"))
    for v in a["verlauf"]:
        teile.append(_absatz(f"{zeit_text(v['created_at'])}: {VERLAUF_TEXT.get(v['art'], v['art'])}"
                             + (f" – {v['text']}" if v.get("text") and v["art"] in ("rueckfrage", "antwort", "storniert") else ""),
                             "ListParagraph", liste=True))
    dokument = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body>'
                + "".join(teile) +
                '<w:sectPr><w:pgSz w:w="11906" w:h="16838"/><w:pgMar w:top="1134" w:right="1134" w:bottom="1134" '
                'w:left="1134" w:header="567" w:footer="567" w:gutter="0"/></w:sectPr></w:body></w:document>')
    jetzt = _jetzt().strftime("%Y-%m-%dT%H:%M:%SZ")
    core = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" '
            'xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dcterms="http://purl.org/dc/terms/" '
            'xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">'
            f'<dc:title>{_x(titel)}</dc:title><dc:language>de-DE</dc:language><dc:creator>InkluDocs</dc:creator>'
            f'<dcterms:created xsi:type="dcterms:W3CDTF">{jetzt}</dcterms:created></cp:coreProperties>')
    typen = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
             '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
             '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
             '<Default Extension="xml" ContentType="application/xml"/>'
             '<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
             '<Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>'
             '<Override PartName="/word/numbering.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.numbering+xml"/>'
             '<Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>'
             '</Types>')
    rels = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>'
            '<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>'
            '</Relationships>')
    doc_rels = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
                '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>'
                '<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/numbering" Target="numbering.xml"/>'
                '</Relationships>')
    with zipfile.ZipFile(ziel, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", typen)
        z.writestr("_rels/.rels", rels)
        z.writestr("docProps/core.xml", core)
        z.writestr("word/document.xml", dokument)
        z.writestr("word/styles.xml", _DOCX_STYLES)
        z.writestr("word/numbering.xml", _DOCX_NUMBERING)
        z.writestr("word/_rels/document.xml.rels", doc_rels)
    return titel


VERLAUF_TEXT = {"bestellt": "Bestellt", "uebernommen": "In Bearbeitung", "rueckfrage": "Rückfrage", "antwort": "Antwort",
                "ergebnis": "Ergebnis hochgeladen", "bericht": "Prüfbericht hochgeladen", "geliefert": "Geliefert",
                "storniert": "Storniert", "notiz": "Interne Notiz"}
