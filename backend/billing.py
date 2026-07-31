"""Zentrale Verbrauchs-Zaehlung fuer das Abo-/Credit-System (Etappe 1, 31.07.2026).

Konzept (siehe backend/ABRECHNUNG.md):
- JEDE kostenpflichtige Aktion laeuft ueber genau zwei Funktionen:
    pruefe_kontingent(user_id)  -> darf die Aktion stattfinden?
    verbuche(user_id, quelle, ...) -> Ereignis in usage_events schreiben
- Etappe 1 ZAEHLT NUR (ABO_ENFORCEMENT=off): pruefe_kontingent liefert immer
  erlaubt=True, aber bereits mit echten Zahlen (verbraucht/kontingent/rest).
  Scharfschalten = ABO_ENFORCEMENT=on in der .env — kein Code-Umbau.
- Team-Toepfe (abo_owner_id) kommen in Etappe 2: konto_user_id ist dafuer
  schon im Schema vorgesehen, aktuell immer = user_id.

Grundregeln (Steve/Michael 31.07.2026):
- 1 Credit = 1 Bild-Generierung, egal ueber welchen Weg (Sammellauf,
  Einzel-Neu-Generieren, API, Chatbot).
- Chatbot: Reden ist frei — sobald er einen Alt-Text ERZEUGT oder AENDERT,
  kostet es 1 Credit (schliesst das Umschreib-Schlupfloch).
- Nur ERFOLGREICHE Aktionen werden verbucht (Fehllaeufe kosten nichts).
- Cache-Treffer werden an den Aufrufstellen nicht verbucht (kosten nichts).
- Admins werden GEZAEHLT (Datenlage!), aber nie gesperrt.
- Verbuchen darf die Generierung NIE zum Absturz bringen: alle DB-Fehler
  werden geloggt und geschluckt.
"""

import logging
import os

from database import get_db

log = logging.getLogger("billing")

# Schalter: "off" = nur zaehlen (Etappe 1), "on" = Kontingente durchsetzen.
ABO_ENFORCEMENT = os.environ.get("ABO_ENFORCEMENT", "off").strip().lower() == "on"

# ---------------------------------------------------------------------------
# Preis- und Plan-Konfiguration
# ---------------------------------------------------------------------------
# Diese Zahlen sind der EINZIGE Ort fuer Kontingente/Preise im Code.
# Startwerte = Vorschlag an Michael vom 31.07.2026, jederzeit aenderbar.
# Neue Werkzeuge bekommen hier eine neue Aktions-Zeile — sonst nichts.

AKTIONS_PREISE = {
    # Aktion                          Credits
    "bild_generierung": 1,          # alle Generierungswege
    "alt_text_aenderung_chatbot": 1,  # Chatbot ersetzt/optimiert einen Alt-Text
}

PLAN_KONTINGENTE = {
    # Plan          Credits pro Monat (None = unbegrenzt/individuell)
    "free": 20,
    "pro": 150,
    "team": 500,
    "enterprise": None,
}

GUELTIGE_QUELLEN = ("sammellauf", "einzeln", "api", "chatbot")


def _konto_fuer(user_id: int):
    """Ermittelt das Abrechnungs-Konto (Etappe 2: Team-Inhaber via
    abo_owner_id). Etappe 1: jeder zahlt auf sein eigenes Konto."""
    return user_id


def monats_verbrauch(konto_user_id: int) -> int:
    """Summe der Credits des laufenden Kalendermonats (UTC) fuer ein Konto."""
    conn = get_db()
    try:
        row = conn.execute(
            "SELECT COALESCE(SUM(credits), 0) FROM usage_events "
            "WHERE konto_user_id = ? AND created_at >= date('now', 'start of month')",
            (konto_user_id,),
        ).fetchone()
        return int(row[0])
    finally:
        conn.close()


def pruefe_kontingent(user_id: int) -> dict:
    """Prueft, ob eine kostenpflichtige Aktion stattfinden darf.

    Rueckgabe immer ein dict:
      erlaubt     True/False (bei ABO_ENFORCEMENT=off IMMER True)
      grund       '' | 'kontingent_erschoepft'
      plan        Plan-Name des Abrechnungs-Kontos
      kontingent  Monats-Credits laut Plan (None = unbegrenzt)
      verbraucht  Credits im laufenden Monat
      rest        verbleibende Credits (None = unbegrenzt)

    Wirft NIE eine Exception nach oben — im Zweifel wird erlaubt
    (Verfuegbarkeit schlaegt Abrechnung, Fehler landet im Log).
    """
    ergebnis = {
        "erlaubt": True, "grund": "", "plan": "free",
        "kontingent": PLAN_KONTINGENTE["free"], "verbraucht": 0, "rest": None,
    }
    try:
        konto_id = _konto_fuer(user_id)
        conn = get_db()
        try:
            row = conn.execute(
                "SELECT COALESCE(plan, 'free') AS plan, is_admin FROM users WHERE id = ?",
                (konto_id,),
            ).fetchone()
        finally:
            conn.close()
        if row is None:
            return ergebnis
        plan = row["plan"] if row["plan"] in PLAN_KONTINGENTE else "free"
        kontingent = PLAN_KONTINGENTE[plan]
        verbraucht = monats_verbrauch(konto_id)
        ergebnis.update({
            "plan": plan,
            "kontingent": kontingent,
            "verbraucht": verbraucht,
            "rest": None if kontingent is None else max(0, kontingent - verbraucht),
        })
        # Admins werden nie gesperrt; ohne Enforcement wird nie gesperrt.
        if not ABO_ENFORCEMENT or row["is_admin"]:
            return ergebnis
        if kontingent is not None and verbraucht >= kontingent:
            ergebnis["erlaubt"] = False
            ergebnis["grund"] = "kontingent_erschoepft"
        return ergebnis
    except Exception:
        log.exception("pruefe_kontingent fehlgeschlagen — Aktion wird erlaubt")
        return ergebnis


def verbuche(user_id: int, quelle: str, aktion: str = "bild_generierung",
             image_id=None, credits=None) -> None:
    """Schreibt EIN Verbrauchs-Ereignis. Nur nach ERFOLGREICHER Aktion rufen.

    credits=None -> Preis laut AKTIONS_PREISE. Fehler werden geloggt und
    geschluckt: die Abrechnung darf die eigentliche Funktion nie brechen.
    """
    try:
        if quelle not in GUELTIGE_QUELLEN:
            log.warning("verbuche: unbekannte Quelle %r — verbuche trotzdem", quelle)
        betrag = AKTIONS_PREISE.get(aktion, 1) if credits is None else int(credits)
        conn = get_db()
        try:
            conn.execute(
                "INSERT INTO usage_events (user_id, konto_user_id, quelle, aktion, credits, image_id) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (user_id, _konto_fuer(user_id), quelle, aktion, betrag, image_id),
            )
            conn.commit()
        finally:
            conn.close()
    except Exception:
        log.exception("verbuche fehlgeschlagen (user=%s quelle=%s aktion=%s image=%s)",
                      user_id, quelle, aktion, image_id)
