"""Zentrale Verbrauchs-Zaehlung fuer das Abo-/Credit-System (Etappe 2, 31.07.2026).

Konzept (siehe backend/ABRECHNUNG.md):
- JEDE kostenpflichtige Aktion laeuft ueber genau zwei Funktionen:
    pruefe_kontingent(user_id)  -> darf die Aktion stattfinden?
    verbuche(user_id, quelle, ...) -> Ereignis in usage_events schreiben
- Etappe 1 ZAEHLTE NUR; der Schalter ist geblieben: bei ABO_ENFORCEMENT=off
  liefert pruefe_kontingent immer erlaubt=True, aber bereits mit echten
  Zahlen. Scharfschalten = ABO_ENFORCEMENT=on in der .env — kein Code-Umbau.

Neues Abomodell (06.08.2026, Steve+Michael): Stufen Free/Single/Team/
Enterprise, feste Laufzeiten 3/6/12 Monate, Auto-Verlaengerung.
- Topf-Aufloesung: users.aktiver_topf zeigt auf den Inhaber des Teams, aus
  dessen Topf das Konto GERADE arbeitet (Kontext-Umschalter auf der
  Abo-Seite; NULL = eigenes Konto). Mitgliedschaften liegen ENTKOPPELT vom
  eigenen Plan in team_mitgliedschaften — ein Konto kann einen eigenen
  Bezahl-Plan haben UND in mehreren Teams Mitglied sein. _konto_fuer
  validiert den Kontext bei jeder Buchung (Mitgliedschaft vorhanden,
  Inhaber-Plan aktiv) und faellt sonst still aufs eigene Konto zurueck.
- Uebertrag ("Rollover"): max. EIN Monatskontingent wandert in den
  Folgemonat, nur fuer Bezahl-Plaene (Free nie). Wird ZUSTANDSLOS aus der
  usage_events-Historie berechnet (siehe _uebertrag) — keine eigene
  Kontostand-Tabelle, die driften koennte.
- Zusatz-Pakete (quota_pakete): gekaufte oder geschenkte Credits mit
  Verfallsdatum (+12 Monate). Verbrauchsreihenfolge: erst das
  Monats-Budget (Kontingent + Uebertrag), dann Pakete — Paket mit dem
  fruehesten Verfallsdatum zuerst (nichts verfallen lassen).

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

import calendar
import logging
import os
from datetime import datetime, timezone

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
    # Michaels Modell (03.08.2026): "Der Export in PDF 5 Credits." Gezaehlt
    # wird PRO EXPORT-VORGANG (ein Klick = 5, auch beim Alle-Dokumente-ZIP —
    # kundenfreundliche Lesart; pro-Dokument waere die strengere, mit Michael
    # klaeren falls gewuenscht). Free-Konten werden bis zur Wasserzeichen-
    # Runde (Umbau-Punkt 3) bewusst NICHT verbucht.
    "pdf_export": 5,            # Grundpreis; tatsaechlich export_preis(anzahl) = 5 + 1 je angefangene 10 (28.08.2026)
    # Word-Werkzeug (26.08.2026): Export der Word-Datei mit Alt-Texten — gleiche
    # Regel wie PDF (pro Vorgang, nur Bezahl-Konten).
    "docx_export": 5,
    # Quickinfo-Werkzeug (27.08.2026): PDF-Formular mit Quickinfos — gleiche
    # Regel wie PDF-Export (pro Vorgang, nur Bezahl-Konten). Die CSV-Feldliste
    # ist kostenlos (kein Schreibvorgang in die PDF).
    "formular_export": 5,
    # Quickinfo-Werkzeug Stufe 2 (27.08.2026): KI-Vorschlaege — EIN Aufruf je
    # Formularseite (reiner Text, ohne Bild), 1 Credit je Seite; Einzel-"Neu
    # generieren" ebenfalls 1. Endgueltige Preisregel mit Michael (offen).
    "quickinfo_generierung": 1,
    "quickinfo_aenderung_chatbot": 1,  # InkluAgent aendert eine Quickinfo (28.08.2026)
}

# Zusatzpakete (Michaels Preise, bestaetigt 03.08.2026). Reine Konfiguration
# fuer Preisseite und Online-Zahlung — der Kauf selbst kommt mit Stripe;
# bis dahin legen Admins Pakete ueber den Rechnungsweg an.
PAKET_PREISE = {
    100: 20.00,
    500: 87.50,
    1000: 150.00,
}

# Abo-Stufen (Steve+Michael, 06.08.2026 — loest das Lizenzschluessel-Modell
# ab): Preise pro Monat. Gebucht wird online wahlweise als Monatsabo
# (seit 19.08.2026, Steves Vorgabe) oder fest fuer 3/6/12 Monate; der
# Rechnungsweg (Michael/Actino) bleibt bei festen Laufzeiten.
# Team/Enterprise: EIN gemeinsamer Credit-Topf, Sitze INKLUSIVE Inhaber,
# Einladungen erhoehen das Kontingent NIE (mehr Credits = Zusatzpakete).
PLAN_PREISE_EUR = {
    "single": 9.95,
    "team": 19.95,
    "enterprise": 49.95,   # Rabatt bereits eingerechnet
}
PLAN_LAUFZEITEN = (1, 3, 6, 12)          # online buchbar; 1 = Monatsabo

# Monatsabo (Steve, 19.08.2026): rund 20 % Aufschlag gegenueber dem
# Laufzeitpreis, NUR online ueber Stripe buchbar, monatlich kuendbar.
PLAN_PREISE_MONATLICH_EUR = {
    "single": 11.95,
    "team": 23.95,
    "enterprise": 59.95,
}
PLAN_LAUFZEITEN_RECHNUNG = (3, 6, 12)    # Admin-/Rechnungsweg (Michael)


def preis_pro_monat(plan: str, monate) -> float:
    """Monatspreis fuer die gewaehlte Laufzeit (1 = Monatsabo mit Aufschlag)."""
    if int(monate or 0) == 1:
        return PLAN_PREISE_MONATLICH_EUR[plan]
    return PLAN_PREISE_EUR[plan]


def treue_preis_eur(plan: str) -> float:
    """Treuekondition (AGB Ziffer 7): Nach einer festen Online-Laufzeit laeuft
    das Abo als Monatsabo zum monatlichen EFFEKTIVPREIS der gebuchten Laufzeit
    weiter — nicht zum regulaeren Monatsabo-Preis. Weil der Laufzeitpreis fuer
    3/6/12 Monate gleich ist, gibt es pro Plan genau EINEN Treuepreis.
    Umgesetzt ueber Stripe-Subscription-Schedules
    (stripe_zahlung.plane_treue_anschluss); Rechnungsweg unberuehrt."""
    return PLAN_PREISE_EUR[plan]

PLAN_KONTINGENTE = {
    # Plan          Credits pro Monat
    # free: 10 (Michaels Modell) — fuer Firmen-Domains GEBUENDELT: alle
    # Free-Konten derselben Domain teilen sich das Volumen (siehe
    # pruefe_kontingent), Freemailer werden NIE gebuendelt.
    "free": 10,
    # single: strikt an EIN Konto gebunden, kein Teilen, kein Einladen.
    "single": 50,
    "team": 100,
    "enterprise": 275,
}

# Sitz-Deckel INKLUSIVE Inhaber (Single bewusst ohne Eintrag = kein Team).
PLAN_SITZE = {
    "team": 5,
    "enterprise": 25,
}

GUELTIGE_QUELLEN = ("sammellauf", "einzeln", "api", "chatbot", "export")

# EXPORT-STAFFEL (Michael Karbe, 28.08.2026, fuer Bilder UND Felder gleich):
# Grundpreis 5 Credits je Export-Vorgang + 1 Credit je ANGEFANGENE 10 Bilder
# bzw. Felder in der exportierten Datei (beim Alle-Dokumente-ZIP zusammen-
# gezaehlt, ein Grundpreis). Free-Konten zahlen auch (10 Credits = ein Formular
# bis 50 Felder). Kein Wasserzeichen mehr; reicht das Guthaben nicht, wird der
# Export verweigert (export_pruefung). Beide Zahlen hier aendern = ueberall.
EXPORT_GRUNDPREIS = 5
EXPORT_STAFFEL = 10


def export_preis(anzahl: int) -> int:
    """Credits fuer einen Export mit `anzahl` Bildern/Feldern (5 + 1 je angefangene 10)."""
    anzahl = max(0, int(anzahl or 0))
    return EXPORT_GRUNDPREIS + (-(-anzahl // EXPORT_STAFFEL))


def verfuegbare_credits(user_id: int):
    """Guthaben, das fuer eine kostenpflichtige Aktion zur Verfuegung steht:
    Monats-Rest + Zusatz-Pakete; None = unbegrenzt (Enterprise/Admin/Enforcement aus)."""
    z = pruefe_kontingent(user_id)
    if z.get("rest") is None:
        return None
    if not ABO_ENFORCEMENT or _ist_admin(user_id):
        return None
    return int(z.get("rest") or 0) + int(z.get("pakete_rest") or 0)


def _ist_admin(user_id: int) -> bool:
    conn = get_db()
    try:
        r = conn.execute("SELECT is_admin FROM users WHERE id = ?", (user_id,)).fetchone()
        return bool(r and r["is_admin"])
    except Exception:
        return False
    finally:
        conn.close()


def export_pruefung(user_id: int, anzahl: int) -> dict:
    """Preis + Guthaben + Entscheidung fuer einen Export.
    {"anzahl", "preis", "verfuegbar" (None = unbegrenzt), "erlaubt", "fehlend"}"""
    preis = export_preis(anzahl)
    verf = verfuegbare_credits(user_id)
    erlaubt = verf is None or verf >= preis
    return {"anzahl": int(anzahl or 0), "preis": preis, "verfuegbar": verf, "erlaubt": erlaubt,
            "fehlend": 0 if erlaubt else preis - int(verf or 0)}


def sitze_fuer_plan(plan: str):
    """Sitz-Deckel eines Plans inkl. Inhaber; None = Plan kennt keine Sitze."""
    return PLAN_SITZE.get(plan)


def plan_ist_abgelaufen(plan_gueltig_bis) -> bool:
    """Auto-Rueckfall auf Free (Michaels Punkt 8, 03.08.2026), LAZY ausgewertet.

    Kein Cronjob, kein UPDATE: ueberall, wo der Plan eines Kontos gelesen
    wird, gilt ein abgelaufener Bezahl-Plan als 'free'. users.plan bleibt
    unveraendert stehen (Beleg + einfache Reaktivierung bei Verlaengerung).
    Regel: gueltig BIS EINSCHLIESSLICH des gespeicherten Tages (ISO-Datum
    oder -Zeitstempel, UTC); leerer Wert = unbefristet.
    """
    if not plan_gueltig_bis:
        return False
    try:
        return str(plan_gueltig_bis)[:10] < datetime.now(timezone.utc).strftime("%Y-%m-%d")
    except Exception:
        return False


def effektiver_plan(row) -> str:
    """Plan einer users-Zeile nach Ablauf-Pruefung (sqlite3.Row oder dict)."""
    try:
        plan = row["plan"]
    except (KeyError, IndexError):
        plan = None
    plan = plan if plan in PLAN_KONTINGENTE else "free"
    try:
        gueltig_bis = row["plan_gueltig_bis"]
    except (KeyError, IndexError):
        gueltig_bis = None
    if plan != "free" and plan_ist_abgelaufen(gueltig_bis):
        return "free"
    return plan


def _konto_fuer(user_id: int):
    """Ermittelt das Abrechnungs-Konto fuer einen Nutzer.

    Neues Abomodell (06.08.2026): users.aktiver_topf zeigt auf den Inhaber
    des Teams, aus dessen Topf das Konto GERADE arbeitet (NULL = eigenes
    Konto). Der Kontext wird bei JEDER Buchung validiert: die Mitgliedschaft
    muss bestehen und der Inhaber einen aktiven Team-/Enterprise-Plan haben —
    sonst faellt die Buchung still aufs eigene Konto zurueck (ein entfernter
    oder abgelaufener Topf darf nie weiter belastet werden). Bewusst nur EIN
    Level Aufloesung, keine Ketten.
    Fehlerfall (DB nicht erreichbar, Nutzer fehlt): eigenes Konto — die
    Verbuchung bleibt damit in jedem Fall moeglich (Nie-Crashen-Garantie).
    """
    try:
        conn = get_db()
        try:
            row = conn.execute(
                "SELECT aktiver_topf FROM users WHERE id = ?", (user_id,)
            ).fetchone()
            if row is None or not row["aktiver_topf"]:
                return user_id
            topf = int(row["aktiver_topf"])
            if topf == user_id:
                return user_id
            inhaber = conn.execute(
                "SELECT u.plan, u.plan_gueltig_bis FROM team_mitgliedschaften m "
                "JOIN users u ON u.id = m.inhaber_id "
                "WHERE m.inhaber_id = ? AND m.mitglied_id = ?",
                (topf, user_id),
            ).fetchone()
        finally:
            conn.close()
        if inhaber is not None and effektiver_plan(inhaber) in PLAN_SITZE:
            return topf
        return user_id
    except Exception:
        log.exception("_konto_fuer fehlgeschlagen — Fallback auf user_id %s", user_id)
        return user_id


def _abrechnungsbeginn_sql(spalte: str = "u.kontingent_reset_am") -> str:
    """SQL-Ausdruck fuer den Beginn des laufenden Abrechnungszeitraums.

    Normalerweise der Monatserste. Nach einem UPGRADE (Steve 07.08.2026:
    „neue Laufzeit, volle Credits sofort") steht in users.kontingent_reset_am
    der Umstellungs-Zeitpunkt — ab dann zaehlt frisch, damit das volle neue
    Kontingent sofort zur Verfuegung steht. Der Wert wirkt nur im Monat der
    Umstellung: spaetestens am Monatsersten gewinnt wieder der Monatsanfang.
    """
    return (f"MAX(date('now', 'start of month'), COALESCE({spalte}, '0000-01-01'))")


def monats_verbrauch(konto_user_id: int, conn=None) -> int:
    """Credits des laufenden ABRECHNUNGS-Zeitraums fuer ein Konto.

    Das ist der Kalendermonat (UTC) — oder, falls im laufenden Monat ein
    Upgrade stattfand, die Zeit seit dem Upgrade (users.kontingent_reset_am).
    So bekommt der Kunde nach einer Hoeherstufung sofort das VOLLE neue
    Kontingent, statt sich den bereits verbrauchten Teil anrechnen zu lassen.

    conn: optional eine BESTEHENDE Verbindung. Noetig, wenn der Aufrufer
    gerade eine offene Transaktion haelt (BEGIN IMMEDIATE in verbuche) —
    nur so sieht die Abfrage den soeben eingefuegten, noch nicht
    committeten Eintrag, und es blockiert nichts gegeneinander.
    """
    eigene_conn = conn is None
    if eigene_conn:
        conn = get_db()
    try:
        row = conn.execute(
            "SELECT COALESCE(SUM(e.credits), 0) FROM usage_events e "
            "LEFT JOIN users u ON u.id = e.konto_user_id "
            f"WHERE e.konto_user_id = ? AND e.created_at >= {_abrechnungsbeginn_sql()}",
            (konto_user_id,),
        ).fetchone()
        return int(row[0])
    finally:
        if eigene_conn:
            conn.close()


def _domain_monats_verbrauch(domain: str, conn=None) -> int:
    """Gemeinsamer Free-Monatsverbrauch ALLER Konten einer Firmen-Domain.

    Punkt 2 des Abo-Umbaus (Michaels Modell, 04.08.2026): "Es koennen sich
    gerne 5 Mitarbeiter oder mehr kostenlos anmelden, in der Summe gibt es
    aber nur das definierte Volumen." Gezaehlt werden nur Konten, die selbst
    auf Free stehen; Betreiber-Konten (Admins) bleiben aussen vor — ihr
    Eigenverbrauch soll keinem Kunden das Free-Volumen leeren. Ereignisse, die
    ein Konto im TEAM-Kontext erzeugt hat, zaehlen ohnehin nicht mit — sie
    stehen auf dem konto_user_id des Team-Inhabers, nicht auf dem Free-Konto.
    Domain-Vergleich exakt (substr ab dem @), kein LIKE.
    """
    eigene_conn = conn is None
    if eigene_conn:
        conn = get_db()
    try:
        # Review-Befund 06.08.: Filter auf den EFFEKTIVEN Plan heben — ein
        # abgelaufener Bezahl-Plan (users.plan bleibt beim Lazy-Rueckfall
        # stehen) zaehlt als Free und muss in der Domain-Summe mitzaehlen,
        # sonst misst pruefe_kontingent ihn am Pool, den seine eigenen
        # Ereignisse nie fuellen (Gratis-Loch nach jeder Kuendigung).
        row = conn.execute(
            "SELECT COALESCE(SUM(e.credits), 0) FROM usage_events e "
            "JOIN users u ON u.id = e.konto_user_id "
            "WHERE substr(u.email, instr(u.email, '@') + 1) = ? "
            "AND (COALESCE(u.plan, 'free') = 'free' "
            "     OR (u.plan_gueltig_bis IS NOT NULL AND date(u.plan_gueltig_bis) < date('now'))) "
            "AND u.is_admin = 0 "
            # Review-Befund 10 (07.08.): HIER bewusst fest der Monatsanfang.
            # Der gemeinsame Free-Topf darf nicht aus Zeitfenstern
            # unterschiedlicher Laenge zusammengerechnet werden — ein
            # Upgrade-Reset an EINEM Konto wuerde sonst der ganzen Domain
            # zusaetzliche Gratis-Credits verschaffen.
            "AND e.created_at >= date('now', 'start of month') ",
            (domain,),
        ).fetchone()
        return int(row[0])
    finally:
        if eigene_conn:
            conn.close()


def _monatsende_iso() -> str:
    """Letzter Tag des laufenden Kalendermonats (UTC) als ISO-Datum.

    Fuer die Anzeige "Kontingent gilt bis ..." — bewusst UTC, weil auch
    monats_verbrauch/usage_events mit UTC-Zeitstempeln arbeiten.
    """
    jetzt = datetime.now(timezone.utc)
    letzter_tag = calendar.monthrange(jetzt.year, jetzt.month)[1]
    return f"{jetzt.year:04d}-{jetzt.month:02d}-{letzter_tag:02d}"


def _uebertrag(konto_id: int, plan: str, kontingent, conn=None) -> int:
    """Uebertrag ("Rollover") ins laufende Monatsbudget, ZUSTANDSLOS berechnet.

    Fachregel (Steve/Michael 31.07.2026): maximal EIN Monatskontingent darf
    in den Folgemonat wandern — nur Bezahl-Plaene, Free nie. Statt einen
    Kontostand zu speichern (der bei Nachbuchungen/Korrekturen driften
    wuerde), wird die Rekurrenz bei jedem Aufruf ueber die Monats-Verbraeuche
    aus usage_events nachgerechnet:

        u_0    = 0                                  (erster Ereignis-Monat)
        u_next = clamp(K + u - verbrauch_monat, 0, K)

    Der Deckel steckt im clamp auf K: mehr als ein volles Monatskontingent
    kann nie mitgenommen werden. Monate OHNE Ereignisse zaehlen als
    Verbrauch 0 (voll gespart), der LAUFENDE Monat ist ausgenommen — sein
    Rest wird erst im Folgemonat zum Uebertrag.

    BEWUSSTE NAEHERUNG: Es wird das AKTUELLE Plan-Kontingent K fuer die
    gesamte Historie angesetzt. Bei Planwechseln (z.B. Upgrade Free -> Pro)
    ist das nicht exakt, haelt die Rechnung aber deterministisch und ohne
    Plan-Historien-Tabelle; Fehlrichtung ist fuer den Kunden meist guenstig.

    conn: optional eine bestehende Verbindung (siehe monats_verbrauch).
    """
    if plan == "free" or not kontingent:
        return 0
    # Review-Befund 3 (07.08.): Nach einem Kontingent-Neustart im laufenden
    # Monat gibt es KEINEN Uebertrag mehr — "frischer Topf" heisst frisch.
    # Sonst wuerde der Uebertrag aus den Vormonaten ein zweites Mal gewaehrt
    # (und durch die K-Naeherung sogar mit dem neuen, groesseren Kontingent
    # nachgerechnet).
    eigene_pruef_conn = conn is None
    _c = get_db() if eigene_pruef_conn else conn
    try:
        _r = _c.execute(
            "SELECT 1 FROM users WHERE id = ? AND kontingent_reset_am IS NOT NULL "
            "AND kontingent_reset_am >= date('now', 'start of month')",
            (konto_id,)).fetchone()
    finally:
        if eigene_pruef_conn:
            _c.close()
    if _r:
        return 0
    eigene_conn = conn is None
    if eigene_conn:
        conn = get_db()
    try:
        rows = conn.execute(
            "SELECT strftime('%Y-%m', created_at) AS monat, "
            "COALESCE(SUM(credits), 0) AS verbrauch "
            "FROM usage_events WHERE konto_user_id = ? "
            "AND created_at < date('now', 'start of month') "
            "GROUP BY monat ORDER BY monat",
            (konto_id,),
        ).fetchall()
    finally:
        if eigene_conn:
            conn.close()
    if not rows:
        # Keine Historie vor dem laufenden Monat -> nichts anzusparen.
        return 0
    verbrauch_je_monat = {r["monat"]: int(r["verbrauch"]) for r in rows}
    jahr, monat = int(rows[0]["monat"][:4]), int(rows[0]["monat"][5:7])
    jetzt = datetime.now(timezone.utc)
    u = 0
    # Kalendermonate LUECKENLOS durchlaufen (auch ereignislose Monate),
    # bis exklusive des laufenden Monats.
    while (jahr, monat) < (jetzt.year, jetzt.month):
        v = verbrauch_je_monat.get(f"{jahr:04d}-{monat:02d}", 0)
        u = max(0, min(kontingent, kontingent + u - v))
        monat += 1
        if monat > 12:
            monat, jahr = 1, jahr + 1
    return u


def starte_kontingent_neu(konto_id: int) -> None:
    """Setzt den Abrechnungs-Zeitraum eines Kontos auf JETZT (Upgrade).

    Vorher wird der noch ungedeckte Ueberhang des laufenden Zeitraums ein
    letztes Mal von den Zusatz-Paketen abgebucht (Review-Befund 8,
    07.08.2026) — sonst verschwaende der Reset diese Forderung dauerhaft,
    weil die Differenzrechnung danach bei null beginnt.

    Der Reset wird HOECHSTENS EINMAL pro Kalendermonat gewaehrt
    (Review-Befund 2): sonst koennte man sich ueber eine Kette von
    Upgrades (single/3 -> single/6 -> ... -> enterprise/12) an einem Tag
    mehrfach das volle Kontingent verschaffen.
    Rueckgabe: nichts; Fehler werden geloggt (Abrechnung darf nie crashen).
    """
    try:
        conn = get_db()
        try:
            conn.isolation_level = None
            conn.execute("BEGIN IMMEDIATE")
            try:
                _pakete_abbuchen(conn, konto_id)
            except Exception:
                log.exception("Ueberhang-Abbuchung vor dem Reset fehlgeschlagen (konto=%s)", konto_id)
            conn.execute(
                "UPDATE users SET kontingent_reset_am = datetime('now') WHERE id = ? "
                "AND (kontingent_reset_am IS NULL "
                "     OR kontingent_reset_am < date('now', 'start of month'))",
                (konto_id,))
            conn.execute("COMMIT")
        except Exception:
            try:
                conn.execute("ROLLBACK")
            except Exception:
                pass
            raise
        finally:
            conn.close()
    except Exception:
        log.exception("Kontingent-Neustart fehlgeschlagen (konto=%s)", konto_id)


def pakete_rest(konto_id: int) -> int:
    """Summe der noch nutzbaren Zusatz-Credits.

    Zwei Verfalls-Arten (Steves Regel 24.08.2026, ersetzt das Ruhen):
    - verfaellt_am NULL: GEKAUFTES Paket — verfaellt nie und ist IMMER
      nutzbar, auch ohne Abo im Free-Plan ("bezahlt ist bezahlt": wer
      kuendigt, braucht seine Pakete trotzdem auf).
    - verfaellt_am gesetzt: Datums-Paket (Alt-Modell/Kulanz-Geschenke).
    """
    conn = get_db()
    try:
        row = conn.execute(
            "SELECT COALESCE(SUM(verbleibend), 0) FROM quota_pakete "
            "WHERE user_id = ? AND verbleibend > 0 AND "
            "(verfaellt_am IS NULL OR verfaellt_am > datetime('now'))",
            (konto_id,),
        ).fetchone()
        return int(row[0])
    finally:
        conn.close()


# Grober Kostensatz pro Credit fuer den Admin-Report (08.08.2026). Er ist
# eine SCHAETZUNG, keine Abrechnung: Die tatsaechlichen KI-Kosten fallen bei
# AWS/Mistral pro Modellaufruf an und lassen sich nicht sauber einem einzelnen
# Kunden zuordnen (Cache-Treffer, mehrere Paesse, unterschiedliche Modelle).
# Der Wert stammt aus den Monatsrechnungen geteilt durch die Credits und ist
# per Umgebungsvariable anpassbar, ohne den Code zu aendern.
KOSTEN_PRO_CREDIT_EUR = float(os.environ.get("KOSTEN_PRO_CREDIT_EUR", "0.06"))


def pruefe_kontingent(user_id: int) -> dict:
    """Prueft, ob eine kostenpflichtige Aktion stattfinden darf.

    Rueckgabe immer ein dict:
      erlaubt          True/False (bei ABO_ENFORCEMENT=off IMMER True)
      grund            '' | 'kontingent_erschoepft'
      plan             Plan-Name des Abrechnungs-Kontos
      kontingent       Monats-Credits laut Plan (None = unbegrenzt)
      uebertrag        aus Vormonaten mitgenommene Credits (max. 1 Kontingent)
      verfuegbar_monat kontingent + uebertrag (None = unbegrenzt)
      verbraucht       Credits im laufenden Monat
      rest             max(0, verfuegbar_monat - verbraucht) (None = unbegrenzt)
      pakete_rest      nutzbare Zusatz-Credits aus quota_pakete
      zeitraum_ende    letzter Tag des laufenden Monats (ISO, UTC)

    Gesperrt (erlaubt=False) wird ERST, wenn das Monats-Budget aufgebraucht
    ist UND keine Zusatz-Pakete mehr da sind — und auch dann nur bei
    ABO_ENFORCEMENT=on und nie fuer Admins. Enterprise (Kontingent None)
    ist immer erlaubt.

    Wirft NIE eine Exception nach oben — im Zweifel wird erlaubt
    (Verfuegbarkeit schlaegt Abrechnung, Fehler landet im Log).
    """
    ergebnis = {
        "erlaubt": True, "grund": "", "plan": "free",
        "kontingent": PLAN_KONTINGENTE["free"], "verbraucht": 0, "rest": None,
        "uebertrag": 0, "verfuegbar_monat": PLAN_KONTINGENTE["free"],
        "pakete_rest": 0, "zeitraum_ende": None, "domain_pool": None,
    }
    try:
        ergebnis["zeitraum_ende"] = _monatsende_iso()
        konto_id = _konto_fuer(user_id)
        conn = get_db()
        try:
            row = conn.execute(
                "SELECT COALESCE(plan, 'free') AS plan, plan_gueltig_bis, is_admin, email "
                "FROM users WHERE id = ?",
                (konto_id,),
            ).fetchone()
        finally:
            conn.close()
        if row is None:
            # Review-Befund 4 (31.07.2026): Konto-Zeile fehlt = Datenfehler
            # (typisch: abo_owner_id zeigte auf einen inzwischen geloeschten
            # Team-Inhaber). NICHT pauschal erlauben — sonst waeren alle
            # Mitglieder eines geloeschten Inhabers dauerhaft gratis.
            # Stattdessen auf den Free-Plan des ECHTEN Nutzers zurueckfallen
            # und dessen EIGENEN Verbrauch pruefen. Fehlt sogar die eigene
            # Nutzer-Zeile, bleibt der bisherige Fail-open (Verfuegbarkeit
            # schlaegt Abrechnung).
            konto_id = user_id
            conn = get_db()
            try:
                # 'free' hart als Plan: der Datenfehler soll konservativ
                # bepreist werden, nicht den (womoeglich unbezahlten)
                # eigenen Plan des Mitglieds uebernehmen.
                row = conn.execute(
                    "SELECT 'free' AS plan, NULL AS plan_gueltig_bis, is_admin, email "
                    "FROM users WHERE id = ?",
                    (konto_id,),
                ).fetchone()
            finally:
                conn.close()
            if row is None:
                return ergebnis
        # Auto-Rueckfall (Punkt 8): abgelaufener Bezahl-Plan zaehlt als Free.
        plan = effektiver_plan(row)
        kontingent = PLAN_KONTINGENTE[plan]
        verbraucht = monats_verbrauch(konto_id)
        # Domain-Buendelung Free (Punkt 2, 04.08.2026): Konten einer
        # Firmen-Domain teilen sich das Free-Volumen — der ANGEZEIGTE und
        # fuer die Sperre massgebliche Verbrauch ist dann die Domain-Summe.
        # Freemailer werden nie gebuendelt (sonst teilten sich alle
        # Gmail-Nutzer der Welt einen Topf), Betreiber-Konten auch nicht.
        # Zusatz-Pakete bleiben bewusst PRO KONTO (gekauft ist gekauft).
        domain_pool = None
        if plan == "free" and not row["is_admin"]:
            _dom = (row["email"] or "").rsplit("@", 1)[-1].strip().lower()
            if _dom and "." in _dom and not ist_freemail_domain(_dom):
                domain_pool = _dom
                verbraucht = _domain_monats_verbrauch(_dom)
        uebertrag = _uebertrag(konto_id, plan, kontingent)
        pakete = pakete_rest(konto_id)
        verfuegbar = None if kontingent is None else kontingent + uebertrag
        ergebnis.update({
            "plan": plan,
            "kontingent": kontingent,
            "uebertrag": uebertrag,
            "verfuegbar_monat": verfuegbar,
            "verbraucht": verbraucht,
            "rest": None if verfuegbar is None else max(0, verfuegbar - verbraucht),
            "pakete_rest": pakete,
            "domain_pool": domain_pool,
        })
        # Admins werden nie gesperrt; ohne Enforcement wird nie gesperrt.
        if not ABO_ENFORCEMENT or row["is_admin"]:
            return ergebnis
        # Reihenfolge der Toepfe: erst Monats-Budget, dann Zusatz-Pakete.
        # Solange noch Paket-Credits da sind, bleibt die Aktion erlaubt.
        if verfuegbar is not None and verbraucht >= verfuegbar and pakete == 0:
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
        konto_id = _konto_fuer(user_id)
        # Review-Befund 2 (31.07.2026): Ereignis-INSERT und Paket-Abbuchung
        # laufen in EINER Transaktion (BEGIN IMMEDIATE) auf EINER Verbindung.
        # BEGIN IMMEDIATE serialisiert alle Schreiber — zwei parallele
        # verbuche-Aufrufe am Budget-Rand koennen sich so weder doppelt
        # abbuchen noch gegenseitig den Ueberhang verschlucken.
        conn = get_db()
        try:
            # isolation_level=None: sonst wuerde das sqlite3-Modul selbst ein
            # BEGIN absetzen und mit unserem BEGIN IMMEDIATE kollidieren.
            conn.isolation_level = None
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                "INSERT INTO usage_events (user_id, konto_user_id, quelle, aktion, credits, image_id) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (user_id, konto_id, quelle, aktion, betrag, image_id),
            )
            # Etappe 2: Liegt der Monats-Verbrauch jetzt UEBER dem Monats-
            # Budget (Kontingent + Uebertrag), wird der Ueberhang von den
            # Zusatz-Paketen abgebucht — in DERSELBEN Transaktion, damit
            # Ereignis und Abbuchung nur gemeinsam sichtbar werden.
            _pakete_abbuchen(conn, konto_id)
            conn.execute("COMMIT")
        except Exception:
            try:
                conn.execute("ROLLBACK")
            except Exception:
                pass
            raise
        finally:
            conn.close()
    except Exception:
        log.exception("verbuche fehlgeschlagen (user=%s quelle=%s aktion=%s image=%s)",
                      user_id, quelle, aktion, image_id)


def _pakete_abbuchen(conn, konto_id: int) -> None:
    """Bucht den Monats-Ueberhang eines Kontos von den Zusatz-Paketen ab.

    Laeuft IN der offenen Transaktion des Aufrufers (verbuche haelt
    BEGIN IMMEDIATE) — alle Lese- und Schreibzugriffe auf DERSELBEN
    Verbindung, damit zwischen Rechnen und Abbuchen kein zweiter Schreiber
    dazwischenkommt (Review-Befund 2, 31.07.2026).

    Soll-Abbuchung als DIFFERENZ statt je Einzel-Ereignis:

        soll    = max(0, verbraucht_monat - (kontingent + uebertrag))
        bereits = Summe der protokollierten Paket-Abbuchungen im lfd. Monat
        abzug   = max(0, soll - bereits)

    Gewaehlte Variante fuer `bereits`: eigene append-only Protokolltabelle
    paket_abbuchungen (statt Rueckrechnung ueber quota_pakete-Endstaende).
    Grund: quota_pakete kennt keinen Zeitbezug je Abbuchung — (groesse -
    verbleibend) wuerde Abbuchungen verschiedener Monate vermischen. Mit dem
    Protokoll ist die Rechnung pro Monat exakt UND idempotent: ein
    wiederholter Lauf findet seine eigenen Buchungen in `bereits` und bucht
    0 nach. Ein frueher mangels Paketen nicht gedeckter Ueberhang wird beim
    naechsten Ereignis automatisch nachgeholt, weil die Differenz ihn
    weiterhin enthaelt — nichts wird verschluckt.

    Abgebucht wird vom Paket mit dem FRUEHESTEN Verfallsdatum zuerst
    (nichts verfallen lassen), nie unter 0. Plaene ohne Kontingent
    (Enterprise/None) buchen NIE Pakete ab. Fehler steigen zum Aufrufer
    verbuche auf — DORT sitzt die Nie-Crashen-Garantie (Rollback + Log),
    hier darf nichts halb committen.
    """
    row = conn.execute(
        "SELECT COALESCE(plan, 'free') AS plan, plan_gueltig_bis FROM users WHERE id = ?",
        (konto_id,),
    ).fetchone()
    if row is None:
        return
    plan = effektiver_plan(row)
    kontingent = PLAN_KONTINGENTE[plan]
    if kontingent is None:
        return
    budget = kontingent + _uebertrag(konto_id, plan, kontingent, conn=conn)
    verbraucht = monats_verbrauch(konto_id, conn=conn)
    soll = max(0, verbraucht - budget)
    # Gleicher Zeitraum wie monats_verbrauch (sonst wuerde nach einem
    # Upgrade-Reset ein alter Ueberhang erneut von den Paketen abgebucht).
    bereits = int(conn.execute(
        "SELECT COALESCE(SUM(a.betrag), 0) FROM paket_abbuchungen a "
        "LEFT JOIN users u ON u.id = a.konto_user_id "
        f"WHERE a.konto_user_id = ? AND a.created_at >= {_abrechnungsbeginn_sql()}",
        (konto_id,),
    ).fetchone()[0])
    abzug_gesamt = soll - bereits
    if abzug_gesamt <= 0:
        return
    # Gekaufte Pakete (verfaellt_am NULL) sind IMMER nutzbar — auch im
    # Free-Plan nach einer Kuendigung (Steves Regel 24.08.2026, ersetzt
    # das fruehere Ruhen). Datums-Pakete zuerst (frueheste zuerst, damit
    # nichts unnoetig verfaellt), unbefristete zuletzt.
    pakete = conn.execute(
        "SELECT id, verbleibend FROM quota_pakete "
        "WHERE user_id = ? AND verbleibend > 0 AND "
        "(verfaellt_am IS NULL OR verfaellt_am > datetime('now')) "
        "ORDER BY (verfaellt_am IS NULL), verfaellt_am, id",
        (konto_id,),
    ).fetchall()
    for paket in pakete:
        if abzug_gesamt <= 0:
            break
        teil = min(int(paket["verbleibend"]), abzug_gesamt)
        conn.execute(
            "UPDATE quota_pakete SET verbleibend = verbleibend - ? WHERE id = ?",
            (teil, paket["id"]),
        )
        conn.execute(
            "INSERT INTO paket_abbuchungen (paket_id, konto_user_id, betrag) "
            "VALUES (?, ?, ?)",
            (paket["id"], konto_id, teil),
        )
        abzug_gesamt -= teil
    if abzug_gesamt > 0:
        # Keine (weiteren) Deckungs-Pakete: nichts geht verloren —
        # usage_events bleibt das vollstaendige Protokoll, die Sperre
        # uebernimmt pruefe_kontingent, und die Differenz-Rechnung holt den
        # Rest nach, sobald wieder Pakete da sind.
        log.info("_pakete_abbuchen: %s Credits Ueberhang ohne Paket-Deckung (konto=%s)",
                 abzug_gesamt, konto_id)


# ---------------------------------------------------------------------------
# Domain-Listen (Free-Buendelung + Wegwerf-Schutz)
# ---------------------------------------------------------------------------

# Freemail-Domains werden bei der Free-Domain-Buendelung NIE gebuendelt
# (sonst teilten sich alle Gmail-Nutzer der Welt einen Topf). Stammt aus dem
# verworfenen Lizenzschluessel-Modell (03.–06.08.2026), bleibt aber die
# Grundlage des Buendelungs-Filters.
FREEMAIL_DOMAINS = frozenset({
    "gmail.com", "googlemail.com", "web.de", "gmx.de", "gmx.net", "gmx.at",
    "gmx.ch", "yahoo.com", "yahoo.de", "outlook.com", "outlook.de",
    "hotmail.com", "hotmail.de", "live.com", "live.de", "icloud.com",
    "me.com", "t-online.de", "aol.com", "mail.de", "mail.com", "freenet.de",
    "posteo.de", "protonmail.com", "proton.me", "tutanota.com", "tuta.io",
    "arcor.de", "online.de", "email.de", "1und1.de", "magenta.de",
})

# Wegwerf-Mail-Anbieter (Punkt 2, 04.08.2026): Registrierung lehnt diese
# Domains ab — sie wuerden sowohl die Free-Domain-Buendelung als auch jedes
# Konten-Limit aushebeln. Liste bewusst kurz und pflegbar (die gaengigsten
# Anbieter); Erweiterung = neue Zeile, kein Umbau.
WEGWERF_DOMAINS = frozenset({
    "mailinator.com", "guerrillamail.com", "guerrillamail.de", "sharklasers.com",
    "grr.la", "10minutemail.com", "10minutemail.de", "temp-mail.org",
    "temp-mail.io", "tempmail.dev", "tempmail.email", "trashmail.com",
    "trash-mail.com", "trashmail.de", "kurzepost.de", "wegwerfemail.de",
    "wegwerfmail.de", "wegwerfmail.net", "byom.de", "yopmail.com",
    "dispostable.com", "mintemail.com", "mohmal.com", "maildrop.cc",
    "getnada.com", "emailondeck.com", "fakemail.net", "throwawaymail.com",
    "mail-temp.com", "moakt.com", "tmpmail.org", "tmpmail.net", "emltmp.com",
})

def ist_freemail_domain(domain: str) -> bool:
    return (domain or "").strip().lower() in FREEMAIL_DOMAINS


def schenke_credits(konto_id: int, menge: int, notiz: str = "", quelle: str = "admin",
                    verfall_monate=12) -> int:
    """Legt ein Zusatz-Credit-Paket fuer ein Abrechnungs-Konto an.

    verfall_monate: Zahl = Datums-Verfall (Alt-Modell/Kulanz-Geschenke, 12
    Monate Vorgabe); None = Michaels Regel 03.08.2026 "Diese Credits gehen
    nicht verloren ... verfallen erst mit der Nicht-Verlaengerung bzw.
    Kuendigung" — gespeichert als verfaellt_am NULL, nutzbar solange das
    Konto einen aktiven Bezahl-Plan hat (siehe pakete_rest). `quelle` ist
    'admin' (Geschenk/Kulanz), 'rechnung' (Kauf ueber Actino) oder spaeter
    'stripe'. Wirft bei ungueltiger Menge einen ValueError — diese Funktion
    laeuft NICHT im Generierungspfad, darf also streng sein.

    Rueckgabe: id der neuen quota_pakete-Zeile.
    """
    menge = int(menge)
    if menge <= 0:
        raise ValueError("menge muss positiv sein")
    conn = get_db()
    try:
        if verfall_monate is None:
            cursor = conn.execute(
                "INSERT INTO quota_pakete (user_id, groesse, verbleibend, quelle, notiz, verfaellt_am) "
                "VALUES (?, ?, ?, ?, ?, NULL)",
                (int(konto_id), menge, menge, quelle, notiz or ""),
            )
        else:
            cursor = conn.execute(
                "INSERT INTO quota_pakete (user_id, groesse, verbleibend, quelle, notiz, verfaellt_am) "
                "VALUES (?, ?, ?, ?, ?, datetime('now', ?))",
                (int(konto_id), menge, menge, quelle, notiz or "",
                 f"+{int(verfall_monate)} months"),
            )
        conn.commit()
        return int(cursor.lastrowid)
    finally:
        conn.close()
