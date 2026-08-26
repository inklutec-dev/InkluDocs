import sqlite3
import os
import hashlib
import secrets
import bcrypt
from datetime import datetime, timedelta

DB_PATH = os.environ.get("INKLUDOCS_DB", "/app/data/inkludocs.db")


def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db():
    conn = get_db()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            display_name TEXT NOT NULL,
            is_admin INTEGER DEFAULT 0,
            is_active INTEGER DEFAULT 1,
            created_at TEXT DEFAULT (datetime('now')),
            last_login TEXT
        );

        CREATE TABLE IF NOT EXISTS password_resets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            token TEXT UNIQUE NOT NULL,
            expires_at TEXT NOT NULL,
            used INTEGER DEFAULT 0,
            created_at TEXT DEFAULT (datetime('now')),
            FOREIGN KEY (user_id) REFERENCES users(id)
        );

        CREATE TABLE IF NOT EXISTS projects (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            filename TEXT NOT NULL,
            original_path TEXT NOT NULL,
            status TEXT DEFAULT 'uploaded',
            total_images INTEGER DEFAULT 0,
            processed_images INTEGER DEFAULT 0,
            created_at TEXT DEFAULT (datetime('now')),
            updated_at TEXT DEFAULT (datetime('now')),
            FOREIGN KEY (user_id) REFERENCES users(id)
        );

        CREATE TABLE IF NOT EXISTS images (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            project_id INTEGER NOT NULL,
            page_number INTEGER NOT NULL,
            image_index INTEGER NOT NULL,
            image_path TEXT NOT NULL,
            image_type TEXT DEFAULT 'unknown',
            alt_text TEXT DEFAULT '',
            alt_text_edited TEXT,
            context_text TEXT DEFAULT '',
            width INTEGER,
            height INTEGER,
            xref INTEGER,
            status TEXT DEFAULT 'pending',
            created_at TEXT DEFAULT (datetime('now')),
            FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS api_keys (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            key_hash TEXT UNIQUE NOT NULL,
            name TEXT NOT NULL,
            created_at TEXT DEFAULT (datetime('now')),
            last_used TEXT,
            is_active INTEGER DEFAULT 1,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        );

        CREATE INDEX IF NOT EXISTS idx_projects_user ON projects(user_id);
        CREATE INDEX IF NOT EXISTS idx_images_project ON images(project_id);
        CREATE INDEX IF NOT EXISTS idx_api_keys_hash ON api_keys(key_hash);
        CREATE TABLE IF NOT EXISTS api_usage (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            api_key_id INTEGER NOT NULL,
            user_id INTEGER NOT NULL,
            timestamp TEXT DEFAULT (datetime('now')),
            processing_time_ms INTEGER,
            model_used TEXT,
            image_size_bytes INTEGER,
            success INTEGER DEFAULT 1,
            error_message TEXT,
            FOREIGN KEY (api_key_id) REFERENCES api_keys(id)
        );

        CREATE INDEX IF NOT EXISTS idx_api_usage_key ON api_usage(api_key_id);
        CREATE INDEX IF NOT EXISTS idx_api_usage_timestamp ON api_usage(timestamp);
        CREATE INDEX IF NOT EXISTS idx_api_usage_user ON api_usage(user_id);

        CREATE TABLE IF NOT EXISTS api_results (
            id TEXT PRIMARY KEY,
            user_id INTEGER NOT NULL,
            api_key_id INTEGER NOT NULL,
            alt_text TEXT NOT NULL DEFAULT '',
            langbeschreibung TEXT NOT NULL DEFAULT '',
            bildtyp TEXT DEFAULT 'unbekannt',
            konfidenz TEXT DEFAULT 'mittel',
            model_used TEXT,
            processing_time_ms INTEGER,
            language TEXT DEFAULT 'de',
            context_text TEXT DEFAULT '',
            created_at TEXT DEFAULT (datetime('now')),
            updated_at TEXT DEFAULT (datetime('now')),
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
            FOREIGN KEY (api_key_id) REFERENCES api_keys(id) ON DELETE CASCADE
        );
        CREATE INDEX IF NOT EXISTS idx_api_results_user ON api_results(user_id);
        CREATE INDEX IF NOT EXISTS idx_api_results_apikey ON api_results(api_key_id);

        CREATE TABLE IF NOT EXISTS email_changes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            new_email TEXT NOT NULL,
            token TEXT UNIQUE NOT NULL,
            expires_at TEXT NOT NULL,
            used INTEGER DEFAULT 0,
            created_at TEXT DEFAULT (datetime('now')),
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS email_verifications (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            email TEXT NOT NULL,
            token TEXT UNIQUE NOT NULL,
            expires_at TEXT NOT NULL,
            used INTEGER DEFAULT 0,
            created_at TEXT DEFAULT (datetime('now')),
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS chat_messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            project_id INTEGER NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            image_refs TEXT,
            intent TEXT,
            created_at TEXT DEFAULT (datetime('now')),
            FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE
        );

        CREATE INDEX IF NOT EXISTS idx_chat_messages_project_id
            ON chat_messages(project_id, created_at);

        -- Multi-Datei Phase 1 (08.06.2026): Dokument-Ebene zwischen Projekt und Bild.
        -- Jede hochgeladene PDF wird zu einem Dokument. So koennen mehrere PDFs in
        -- einem Projekt liegen, ohne dass sich Seitenzahlen ueberschneiden — die
        -- Spalte document_id an images traegt die Aufloesung.
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            project_id INTEGER NOT NULL,
            doc_index INTEGER NOT NULL,
            original_filename TEXT NOT NULL,
            display_name TEXT,
            original_path TEXT NOT NULL DEFAULT '',
            extraction_method TEXT DEFAULT 'fitz',
            total_images INTEGER DEFAULT 0,
            created_at TEXT DEFAULT (datetime('now')),
            FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE
        );
        CREATE INDEX IF NOT EXISTS idx_documents_project ON documents(project_id, doc_index);

        -- Gastzugang / Projekt-Freigabe (19.06.2026): pro Einladung ein Token,
        -- gebunden an EIN Projekt + EINE Gast-E-Mail. Token = Geheimnis, die
        -- E-Mail-Eingabe des Gastes = Bestaetigung. Datenlogik in sharing.py.
        CREATE TABLE IF NOT EXISTS shares (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            project_id INTEGER NOT NULL,
            token TEXT UNIQUE NOT NULL,
            guest_email TEXT NOT NULL,
            status TEXT DEFAULT 'active',
            created_by INTEGER NOT NULL,
            created_at TEXT DEFAULT (datetime('now')),
            expires_at TEXT,
            email_confirmed_at TEXT,
            completed_at TEXT,
            FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE,
            FOREIGN KEY (created_by) REFERENCES users(id)
        );
        CREATE INDEX IF NOT EXISTS idx_shares_token ON shares(token);
        CREATE INDEX IF NOT EXISTS idx_shares_project ON shares(project_id);

        -- Gastzugang-Rollen (10.07.2026): Pruefstatus pro Bild UND Rolle
        -- ('kunde'/'lektorat'). images.review_status bleibt als Spiegel des
        -- jeweils LETZTEN Gast-Urteils fuer Badge + Dashboard-Zaehler erhalten.
        CREATE TABLE IF NOT EXISTS image_reviews (
            image_id INTEGER NOT NULL,
            role TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'offen',
            reviewed_at TEXT,
            PRIMARY KEY (image_id, role),
            FOREIGN KEY (image_id) REFERENCES images(id) ON DELETE CASCADE
        );

        -- Nachrichten (19.06.2026): bewusst ALLGEMEIN gehalten, damit daraus
        -- spaeter ohne Umbau das volle In-App-Messaging (User<->User, Abteilungen)
        -- wachsen kann. v1 nutzt nur Gast<->Besitzer, optional an Bild/Projekt.
        -- sender_user_id ODER sender_guest gesetzt (je nach Absender).
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            project_id INTEGER,
            image_id INTEGER,
            sender_user_id INTEGER,
            sender_guest TEXT,
            recipient_user_id INTEGER,
            body TEXT NOT NULL,
            msg_type TEXT DEFAULT 'message',
            read_at TEXT,
            created_at TEXT DEFAULT (datetime('now')),
            FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE
        );
        CREATE INDEX IF NOT EXISTS idx_messages_project ON messages(project_id);
        CREATE INDEX IF NOT EXISTS idx_messages_image ON messages(image_id);

        -- Eigene Prompts (06.07.2026): vom Nutzer gespeicherte Zusatz-Anweisungen
        -- fuer die Alt-Text-Generierung (Prompt-Verwaltung). Ein Projekt kann ueber
        -- projects.prompt_id (Migration unten) genau einen aktiven Prompt
        -- referenzieren; NULL = kein eigener Prompt (Standardverhalten).
        CREATE TABLE IF NOT EXISTS user_prompts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            name TEXT NOT NULL,
            description TEXT DEFAULT '',
            category TEXT DEFAULT '',
            prompt_text TEXT NOT NULL,
            created_at TEXT DEFAULT (datetime('now')),
            updated_at TEXT DEFAULT (datetime('now')),
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        );
        CREATE INDEX IF NOT EXISTS idx_user_prompts_user ON user_prompts(user_id);
    """)
    conn.commit()

    # Abo-/Credit-System Etappe 1 (31.07.2026): jede kostenpflichtige Aktion
    # wird als einzelnes Ereignis gespeichert (siehe backend/ABRECHNUNG.md).
    # konto_user_id = wessen Kontingent belastet wird (Etappe 2: Team-Inhaber);
    # Etappe 1 immer = user_id.
    conn.execute('''
        CREATE TABLE IF NOT EXISTS usage_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            konto_user_id INTEGER NOT NULL,
            quelle TEXT NOT NULL,
            aktion TEXT NOT NULL DEFAULT 'bild_generierung',
            credits INTEGER NOT NULL DEFAULT 1,
            image_id INTEGER,
            created_at TEXT DEFAULT (datetime('now'))
        )
    ''')
    conn.execute("CREATE INDEX IF NOT EXISTS idx_usage_events_konto_monat ON usage_events(konto_user_id, created_at)")

    # Abo-Etappe 2 (31.07.2026): Zusatz-Credit-Pakete (Geschenk/Kauf/Rechnung).
    # user_id = Abrechnungs-KONTO (bei Teams der zahlende Inhaber, siehe
    # billing._konto_fuer). verbleibend wird beim Verbrauch dekrementiert,
    # verfallene Pakete bleiben als Beleg stehen (nie loeschen).
    # quelle: 'admin' (Geschenk) | 'stripe' (Etappe 3) | 'rechnung'.
    conn.execute('''
        CREATE TABLE IF NOT EXISTS quota_pakete (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            groesse INTEGER NOT NULL,
            verbleibend INTEGER NOT NULL,
            quelle TEXT NOT NULL,
            notiz TEXT DEFAULT '',
            erstellt_am TEXT DEFAULT (datetime('now')),
            verfaellt_am TEXT
        )
    ''')
    conn.execute("CREATE INDEX IF NOT EXISTS idx_quota_pakete_konto ON quota_pakete(user_id, verfaellt_am)")

    # Punkt 4 des Abo-Umbaus (04.08.2026, Michaels Regel): verfaellt_am darf
    # jetzt NULL sein = "verfaellt erst mit der Kuendigung" (Auswertung lazy
    # in billing.pakete_rest). Bestands-DBs tragen noch NOT NULL — SQLite
    # kann das nicht per ALTER entfernen, darum einmaliger idempotenter
    # Tabellen-Umbau (Neuanlage, Kopie, Umbenennung; Daten + ids bleiben).
    _pragma = conn.execute("PRAGMA table_info(quota_pakete)").fetchall()
    _verfall_notnull = any(r[1] == "verfaellt_am" and r[3] == 1 for r in _pragma)
    if _verfall_notnull:
        conn.executescript('''
            CREATE TABLE quota_pakete_neu (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                groesse INTEGER NOT NULL,
                verbleibend INTEGER NOT NULL,
                quelle TEXT NOT NULL,
                notiz TEXT DEFAULT '',
                erstellt_am TEXT DEFAULT (datetime('now')),
                verfaellt_am TEXT
            );
            INSERT INTO quota_pakete_neu (id, user_id, groesse, verbleibend, quelle, notiz, erstellt_am, verfaellt_am)
                SELECT id, user_id, groesse, verbleibend, quelle, notiz, erstellt_am, verfaellt_am FROM quota_pakete;
            DROP TABLE quota_pakete;
            ALTER TABLE quota_pakete_neu RENAME TO quota_pakete;
        ''')
        conn.execute("CREATE INDEX IF NOT EXISTS idx_quota_pakete_konto ON quota_pakete(user_id, verfaellt_am)")
        conn.commit()

    # Abo-Etappe 2, Nachbesserung (31.07.2026, Review-Befund 1): Team-Beitritt
    # nur noch mit ZUSTIMMUNG des Eingeladenen. Eine Einladung ist ein Token
    # mit Ablaufdatum (Muster wie password_resets); erst das Einloesen ueber
    # GET /team-einladung/{token} haengt das Konto um. Vorher konnte ein
    # Team-Inhaber fremde Bestandskonten per stillem UPDATE kapern.
    conn.execute('''
        CREATE TABLE IF NOT EXISTS team_einladungen (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            token TEXT UNIQUE NOT NULL,
            inhaber_id INTEGER NOT NULL,
            email TEXT NOT NULL,
            erstellt_am TEXT DEFAULT (datetime('now')),
            gueltig_bis TEXT NOT NULL,
            eingeloest_am TEXT
        )
    ''')
    conn.execute("CREATE INDEX IF NOT EXISTS idx_team_einladungen_token ON team_einladungen(token)")

    # Abo-Etappe 2, Nachbesserung (31.07.2026, Review-Befund 2): append-only
    # Protokoll der Paket-Abbuchungen. billing._pakete_abbuchen rechnet die
    # Soll-Abbuchung als DIFFERENZ (Monats-Ueberhang minus bereits
    # protokollierte Abbuchungen) — dafuer braucht es eine Summe je Monat,
    # die quota_pakete selbst nicht hergibt (dort steht nur der Endstand
    # ohne Zeitbezug der einzelnen Abbuchungen).
    conn.execute('''
        CREATE TABLE IF NOT EXISTS paket_abbuchungen (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            paket_id INTEGER NOT NULL,
            konto_user_id INTEGER NOT NULL,
            betrag INTEGER NOT NULL,
            created_at TEXT DEFAULT (datetime('now'))
        )
    ''')
    conn.execute("CREATE INDEX IF NOT EXISTS idx_paket_abbuchungen_konto_monat ON paket_abbuchungen(konto_user_id, created_at)")

    # Lizenzschluessel (04.08.2026, Michaels Modell — Mail 03.08., bestaetigt):
    # EIN Schluessel pro Unternehmen statt Pro/Team/Enterprise. Der Schluessel
    # ist an die E-Mail-Domain der Firma gebunden (Missbrauchs-Schutz, Steves
    # Nachschaerfung 3); der ERSTE Aktivierer wird Topf-Inhaber (plan='lizenz'),
    # jeder weitere Aktivierer mit passender Domain haengt sich per
    # users.abo_owner_id in denselben Firmen-Topf — beliebig viele Nutzer.
    # domain NULL = bindet sich bei der Erst-Aktivierung an die Domain des
    # Aktivierers (Freemail-Domains werden dabei abgelehnt).
    # status: 'neu' (erzeugt, noch nie aktiviert) | 'aktiv' | 'gesperrt'.
    # Erzeugt werden Schluessel von Voll-Admins — das ist zugleich Michaels
    # Rechnungsweg (Kauf auf Rechnung ueber Actino, Mail 03.08.).
    conn.execute('''
        CREATE TABLE IF NOT EXISTS lizenzschluessel (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            schluessel TEXT UNIQUE NOT NULL,
            domain TEXT,
            inhaber_user_id INTEGER,
            monats_credits INTEGER NOT NULL DEFAULT 50,
            laufzeit_monate INTEGER NOT NULL DEFAULT 6,
            gueltig_bis TEXT,
            status TEXT NOT NULL DEFAULT 'neu',
            notiz TEXT DEFAULT '',
            erstellt_von INTEGER,
            created_at TEXT DEFAULT (datetime('now')),
            aktiviert_am TEXT
        )
    ''')
    conn.execute("CREATE INDEX IF NOT EXISTS idx_lizenzschluessel_key ON lizenzschluessel(schluessel)")

    # Neues Abomodell (06.08.2026, Steve+Michael): Mitgliedschaft ist vom
    # eigenen Plan ENTKOPPELT. Ein Konto kann einen eigenen Bezahl-Plan haben
    # UND gleichzeitig Mitglied in einem oder mehreren Team-/Enterprise-
    # Toepfen sein — auch in mehreren Teams. Aus welchem Topf verbraucht wird,
    # entscheidet users.aktiver_topf (Kontext-Umschalter auf der Abo-Seite).
    # Ersetzt users.abo_owner_id (Alt-Spalte bleibt leer stehen, wird nirgends
    # mehr gelesen). Die lizenzschluessel-Tabelle oben bleibt als Beleg,
    # das Schluessel-Modell selbst wurde am 06.08.2026 zugunsten der
    # Abo-Stufen Single/Team/Enterprise verworfen (Steve+Michael).
    conn.execute('''
        CREATE TABLE IF NOT EXISTS team_mitgliedschaften (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            inhaber_id INTEGER NOT NULL,
            mitglied_id INTEGER NOT NULL,
            created_at TEXT DEFAULT (datetime('now')),
            UNIQUE(inhaber_id, mitglied_id)
        )
    ''')
    conn.execute("CREATE INDEX IF NOT EXISTS idx_team_mitglied ON team_mitgliedschaften(mitglied_id)")

    # Kleiner Schluessel-Wert-Speicher fuer System-Merker (letzter
    # Abo-Tageslauf, bereits versandte Erinnerungs-Mails) — generisch, damit
    # nicht jede Kleinigkeit eine eigene Tabelle braucht.
    conn.execute('''
        CREATE TABLE IF NOT EXISTS system_kv (
            key TEXT PRIMARY KEY,
            value TEXT,
            updated_at TEXT DEFAULT (datetime('now'))
        )
    ''')

    # Nachweis der Widerrufs-Zustimmung (08.08.2026, Verbraucherrecht):
    # Vor jeder kostenpflichtigen Buchung muss der Kunde ausdruecklich
    # zustimmen, dass wir vor Ablauf der Widerrufsfrist mit der Leistung
    # beginnen (§ 356 Abs. 4/5 BGB). Diese Zustimmung muessen WIR im
    # Streitfall beweisen — darum wird sie mit Zeitpunkt, Textfassung und
    # Absender-Kennung protokolliert und nicht nur "irgendwo angehakt".
    conn.execute('''
        CREATE TABLE IF NOT EXISTS widerruf_zustimmungen (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            plan TEXT NOT NULL,
            laufzeit_monate INTEGER NOT NULL,
            summe_cent INTEGER NOT NULL,
            belehrung_fassung TEXT NOT NULL,
            absender TEXT DEFAULT '',
            created_at TEXT DEFAULT (datetime('now'))
        )
    ''')

    # Kuendigungen ueber den gesetzlichen Kuendigungsknopf (§ 312k BGB).
    # Der Zeitpunkt des ZUGANGS ist rechtlich entscheidend und wird darum
    # unabhaengig davon festgehalten, ob sich die Erklaerung automatisch
    # einem Konto zuordnen liess.
    conn.execute('''
        CREATE TABLE IF NOT EXISTS kuendigungen (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT NOT NULL,
            name TEXT DEFAULT '',
            art TEXT NOT NULL DEFAULT 'ordentlich',
            grund TEXT DEFAULT '',
            zusatz TEXT DEFAULT '',
            user_id INTEGER,
            vollzogen INTEGER NOT NULL DEFAULT 0,
            wirksam_zum TEXT,
            absender TEXT DEFAULT '',
            created_at TEXT DEFAULT (datetime('now'))
        )
    ''')

    # Backward-compatible migrations using ALTER TABLE with try/except
    _migrate_columns(conn)

    conn.close()

    # T6 (03.05.2026, Phase A): Persistente Cache-Tabelle anlegen.
    # Eigene Connection in cache.init_schema(), idempotent (CREATE TABLE IF NOT EXISTS).
    from cache import init_schema as _cache_init_schema
    _cache_init_schema()


def _migrate_columns(conn):
    """Add new columns to existing tables. Uses try/except for idempotency."""
    migrations = [
        # Abo-/Credit-System Etappe 1 (31.07.2026)
        ("users", "plan", "ALTER TABLE users ADD COLUMN plan TEXT DEFAULT 'free'"),
        # Abo-Etappe 2 (31.07.2026): Team-Toepfe — Mitglieder zeigen auf den
        # zahlenden Inhaber (NULL = eigenes Konto). Genau EIN Level, keine Ketten.
        ("users", "abo_owner_id", "ALTER TABLE users ADD COLUMN abo_owner_id INTEGER"),
        # Abo-Etappe 2: Ablaufdatum fuer Rechnungskunden/Gruender-Tarif
        # (NULL = unbefristet; Auswertung folgt mit dem Rechnungs-Workflow).
        ("users", "plan_gueltig_bis", "ALTER TABLE users ADD COLUMN plan_gueltig_bis TEXT"),
        # Treue-Monatsabo (AGB Ziffer 7, 24.08.2026): 1 = das Abo laeuft nach
        # einer festen Laufzeit in der Anschluss-Kondition weiter.
        ("users", "plan_treue", "ALTER TABLE users ADD COLUMN plan_treue INTEGER DEFAULT 0"),
        # Vector graphics support (from earlier migration)
        ("images", "bbox_x0", "ALTER TABLE images ADD COLUMN bbox_x0 REAL"),
        ("images", "bbox_y0", "ALTER TABLE images ADD COLUMN bbox_y0 REAL"),
        ("images", "bbox_x1", "ALTER TABLE images ADD COLUMN bbox_x1 REAL"),
        ("images", "bbox_y1", "ALTER TABLE images ADD COLUMN bbox_y1 REAL"),
        ("images", "is_vector", "ALTER TABLE images ADD COLUMN is_vector INTEGER DEFAULT 0"),
        ("images", "konfidenz", "ALTER TABLE images ADD COLUMN konfidenz TEXT DEFAULT 'mittel'"),
        # New columns for this refactoring
        ("projects", "project_type", "ALTER TABLE projects ADD COLUMN project_type TEXT DEFAULT 'pdf'"),
        ("projects", "source_url", "ALTER TABLE projects ADD COLUMN source_url TEXT"),
        # Ausgabesprache der generierten Alt-Texte (03.07.2026, alirodocs):
        # lebende Projekt-Einstellung wie use_context — gilt fuer alles ab jetzt Generierte.
        ("projects", "alt_language", "ALTER TABLE projects ADD COLUMN alt_language TEXT DEFAULT 'de'"),
        ("projects", "prompt_id", "ALTER TABLE projects ADD COLUMN prompt_id INTEGER"),
        # Sprache, in der der Alt-Text dieses Bildes GENERIERT wurde (fuer die
        # Vorlese-Stimme; NULL = deutscher Altbestand). Gemischte Projekte moeglich.
        ("images", "gen_language", "ALTER TABLE images ADD COLUMN gen_language TEXT"),
        ("images", "langbeschreibung", "ALTER TABLE images ADD COLUMN langbeschreibung TEXT DEFAULT ''"),
        ("images", "original_alt", "ALTER TABLE images ADD COLUMN original_alt TEXT DEFAULT ''"),
        ("images", "feedback", "ALTER TABLE images ADD COLUMN feedback TEXT DEFAULT ''"),
        ("users", "email_verified", "ALTER TABLE users ADD COLUMN email_verified INTEGER DEFAULT 1"),
        ("users", "news_seen_until", "ALTER TABLE users ADD COLUMN news_seen_until TEXT"),
        # Geteilte-Projekte-Kasten (15.07.2026): Lese-Stand wie news_seen_until,
        # aber als UTC-Zeitstempel (vergleichbar mit created_at/reviewed_at/completed_at).
        ("users", "reviews_seen_until", "ALTER TABLE users ADD COLUMN reviews_seen_until TEXT"),
        # Admin-Rechte-Stufe (14.06.2026): 'full' = alle Rechte, 'view' = nur Einsicht.
        # Default 'full', damit bestehende Admins (Steve, Michael) volle Rechte behalten.
        ("users", "admin_level", "ALTER TABLE users ADD COLUMN admin_level TEXT DEFAULT 'full'"),
        # v3.3 (14.04.2026): Dreistufige Pipeline mit Validator
        ("images", "needs_review", "ALTER TABLE images ADD COLUMN needs_review INTEGER DEFAULT 0"),
        ("images", "pipeline_steps", "ALTER TABLE images ADD COLUMN pipeline_steps TEXT DEFAULT ''"),
        ("images", "validation_result", "ALTER TABLE images ADD COLUMN validation_result TEXT DEFAULT ''"),
        # PDFIX-INTEGRATION (24.04.2026): Extraktionsweg pro Projekt festhalten
        ("projects", "extraction_method", "ALTER TABLE projects ADD COLUMN extraction_method TEXT DEFAULT 'fitz'"),
        # Seitenansicht-Feature (27.05.2026): PNG der ganzen Seite + Volltext pro Bild
        ("images", "page_view_path", "ALTER TABLE images ADD COLUMN page_view_path TEXT DEFAULT ''"),
        ("images", "page_text", "ALTER TABLE images ADD COLUMN page_text TEXT DEFAULT ''"),
        # 19.06.2026: KI-Kontext-Schalter pro Projekt + Vermerk pro Bild.
        # use_context Default 1 (=an) -> bestehende Projekte behalten ihr Verhalten.
        # context_mode ('mit'/'ohne'/'') ist NUR Anzeige, wird NIE exportiert.
        ("projects", "use_context", "ALTER TABLE projects ADD COLUMN use_context INTEGER DEFAULT 1"),
        ("images", "context_mode", "ALTER TABLE images ADD COLUMN context_mode TEXT DEFAULT ''"),
        # Dashboard (02.06.2026): freier Projektname + Werkzeug-Zuordnung
        ("projects", "name", "ALTER TABLE projects ADD COLUMN name TEXT DEFAULT ''"),
        ("projects", "tool", "ALTER TABLE projects ADD COLUMN tool TEXT DEFAULT 'alttext'"),
        # Multi-Datei Phase 1 (08.06.2026): Bild -> Dokument-Zuordnung.
        # NULL-Wert beim ersten Lesen ist normal — der Backfill weiter unten
        # legt fuer bestehende PDF-Projekte je ein Dokument 1 an und setzt
        # die Spalte. Neue Uploads tragen den Wert direkt beim INSERT.
        ("images", "document_id", "ALTER TABLE images ADD COLUMN document_id INTEGER"),
        ("shares", "guest_name", "ALTER TABLE shares ADD COLUMN guest_name TEXT DEFAULT ''"),
        # Gastzugang / Review (19.06.2026): Pruefstatus pro Bild. 'offen' = noch nicht
        # geprueft (Default), 'freigegeben' = ok, 'zu_ueberarbeiten' = Aenderung gewuenscht.
        # Der eigentliche Kommentar liegt in der messages-Tabelle (msg_type review_note),
        # nicht hier -> Status bleibt schlank abfragbar fuer Badge + Dashboard-Zaehler.
        ("images", "review_status", "ALTER TABLE images ADD COLUMN review_status TEXT DEFAULT 'offen'"),
        ("images", "reviewed_at", "ALTER TABLE images ADD COLUMN reviewed_at TEXT"),
        # UI-Sprache pro User (01.07.2026): ISO-639-1 - 'de'/'en'/'fr'/'es'.
        # Default 'de', damit bestehende User Deutsch behalten. Steuert NUR die
        # Oberflaechen-Sprache, NICHT die Alt-Text-Ausgabesprache (separates Feature).
        ("users", "language", "ALTER TABLE users ADD COLUMN language TEXT DEFAULT 'de'"),
        # Gastzugang-Rollen (10.07.2026): Rolle der Einladung — 'kunde' (Endkunde,
        # Default, entspricht dem bisherigen Verhalten) oder 'lektorat'.
        ("shares", "role", "ALTER TABLE shares ADD COLUMN role TEXT DEFAULT 'kunde'"),
        # Rollen-Workflow Etappe 4 (15.07.2026): Nachrichten-Verlauf pro Bild
        # (msg_type 'chat'). Absender-Rolle ('besitzer'/'lektorat'/'kunde') und
        # Anzeigename werden denormalisiert mitgespeichert — der Verlauf bleibt
        # auch nach Widerruf einer Freigabe lesbar zuordenbar.
        ("messages", "sender_role", "ALTER TABLE messages ADD COLUMN sender_role TEXT DEFAULT ''"),
        ("messages", "sender_name", "ALTER TABLE messages ADD COLUMN sender_name TEXT DEFAULT ''"),
        # Neues Abomodell (06.08.2026): aktiver Credit-Topf (NULL = eigener,
        # sonst inhaber_id einer Mitgliedschaft), feste Laufzeit 3/6/12 Monate,
        # Auto-Verlaengerung (Kuendigen = Schalter auf 0, Nutzung bis
        # Laufzeitende), Teamname des Inhabers (Anzeige beim Mitglied).
        ("users", "aktiver_topf", "ALTER TABLE users ADD COLUMN aktiver_topf INTEGER"),
        ("users", "plan_laufzeit_monate", "ALTER TABLE users ADD COLUMN plan_laufzeit_monate INTEGER"),
        ("users", "auto_verlaengerung", "ALTER TABLE users ADD COLUMN auto_verlaengerung INTEGER DEFAULT 1"),
        ("users", "team_name", "ALTER TABLE users ADD COLUMN team_name TEXT DEFAULT ''"),
        # Missbrauchsbremse API (11.08.2026, Steve): Tageslimit der Bild-
        # Generierung pro KONTO einstellbar (NULL = globaler Standard
        # DAILY_IMAGE_LIMIT, 0 = gesperrt). Fuer Betreiber-Konten mit
        # unbegrenzten Credits ist das die EINZIGE Bremse — deshalb
        # einstellbar statt Admin-Ausnahme.
        ("users", "api_tageslimit", "ALTER TABLE users ADD COLUMN api_tageslimit INTEGER"),
        # Sicherheits-Review 06.08.: Einladungen an UNBEKANNTE Adressen legen
        # das Konto zwar an, die Mitgliedschaft entsteht aber erst, wenn die
        # Person ihr Passwort setzt (= registriert). konto_neu=1 kennzeichnet
        # diese Einladungen; do_reset_password loest sie automatisch ein.
        ("team_einladungen", "konto_neu", "ALTER TABLE team_einladungen ADD COLUMN konto_neu INTEGER DEFAULT 0"),
        # Stripe-Anbindung (06.08.2026 nachts): Kunde/Subscription je Konto,
        # plan_quelle ('stripe' | 'rechnung' | NULL=Altbestand) entscheidet,
        # WER verlaengert — Stripe-Abos verlaengert invoice.paid, NIE der
        # Tageslauf (sonst doppelt).
        ("users", "stripe_customer_id", "ALTER TABLE users ADD COLUMN stripe_customer_id TEXT"),
        ("users", "stripe_subscription_id", "ALTER TABLE users ADD COLUMN stripe_subscription_id TEXT"),
        ("users", "plan_quelle", "ALTER TABLE users ADD COLUMN plan_quelle TEXT"),
        # Planwechsel (07.08.2026): Ein DOWNGRADE wird nur vorgemerkt und
        # wirkt zum Ende der bezahlten Laufzeit — bis dahin laeuft alles
        # unveraendert weiter. geplanter_plan/geplante_laufzeit halten den
        # Wunsch; der Tageslauf stellt am Stichtag um (plan_gueltig_bis).
        # Upgrades brauchen das nicht (sofort wirksam).
        ("users", "geplanter_plan", "ALTER TABLE users ADD COLUMN geplanter_plan TEXT"),
        ("users", "geplante_laufzeit", "ALTER TABLE users ADD COLUMN geplante_laufzeit INTEGER"),
        # Review-Befund 2 (07.08.2026): FESTER Stichtag des vorgemerkten
        # Wechsels. plan_gueltig_bis kann sich durch Stripe-Webhooks
        # verschieben — der Wechsel muss trotzdem am vorgemerkten Datum
        # greifen, sonst bleibt er bei Stripe-Abos fuer immer liegen.
        ("users", "geplant_ab", "ALTER TABLE users ADD COLUMN geplant_ab TEXT"),
        # Steve 07.08.2026: Beim UPGRADE startet eine neue Laufzeit und es
        # gibt sofort das VOLLE neue Kontingent. Dieser Zeitstempel markiert
        # den Neustart — billing.monats_verbrauch zaehlt ab hier frisch
        # (wirkt nur im Monat der Umstellung, danach gilt wieder der Monatsanfang).
        ("users", "kontingent_reset_am", "ALTER TABLE users ADD COLUMN kontingent_reset_am TEXT"),
        # Multi-Datei Phase 2 (14.08.2026): Web-Projekte bekommen die Dokument-
        # Ebene (eine gescannte Adresse = eine "Webseite"), Grafik-Projekte
        # Umbenennen/Loeschen pro Bild.
        #  - images.original_filename: Original-Dateiname beim Grafik-Upload
        #    (Anzeige "Bild 1: urlaubsfoto.jpg"); Altbestand bleibt leer.
        #  - images.display_name: vom Nutzer vergebener Anzeigename pro Bild
        #    (NULL = Rueckfall auf original_filename, wie bei documents).
        #  - documents.source_url: Adresse der gescannten Webseite (Link
        #    "Webseite im neuen Tab oeffnen"); bei PDF-Dokumenten leer.
        #  - documents.page_text: Textinhalt der Webseite fuer die
        #    "Seitentext anzeigen"-Klappe. Bei PDFs bleibt die Spalte leer,
        #    dort haengt der Seitentext wie gehabt an images.page_text.
        ("images", "original_filename", "ALTER TABLE images ADD COLUMN original_filename TEXT DEFAULT ''"),
        ("images", "display_name", "ALTER TABLE images ADD COLUMN display_name TEXT"),
        ("documents", "source_url", "ALTER TABLE documents ADD COLUMN source_url TEXT DEFAULT ''"),
        ("documents", "page_text", "ALTER TABLE documents ADD COLUMN page_text TEXT DEFAULT ''"),
        # WORD-WERKZEUG (26.08.2026): Anker des Bildvorkommens in der .docx
        # ("<part>|<docPr-id>", z. B. "word/document.xml|7"). Damit findet
        # docx_export das Element beim Zurueckschreiben wieder. PDF: leer.
        ("images", "docx_anker", "ALTER TABLE images ADD COLUMN docx_anker TEXT DEFAULT ''"),
    ]

    for table, column, sql in migrations:
        try:
            conn.execute(f"SELECT {column} FROM {table} LIMIT 1")
        except Exception:
            try:
                conn.execute(sql)
                print(f"Migration: Added {table}.{column}")
            except Exception as e:
                print(f"Migration warning ({table}.{column}): {e}")

    # Neues Abomodell (06.08.2026): Alt-Daten einmalig ueberfuehren.
    # (a) users.abo_owner_id -> team_mitgliedschaften + aktiver_topf; danach
    #     wird die Alt-Spalte geleert (bleibt bestehen, wird nie mehr gelesen).
    # (b) Alt-Plaene: 'pro' -> 'single'; 'lizenz' -> 'team' (der Firmen-Topf
    #     lebt als Team weiter; betrifft nur Staging-Testdaten — die Prod-DB
    #     kennt diese Plaene noch nicht).
    # Idempotent: INSERT OR IGNORE + WHERE-Bedingungen; zweiter Lauf tut nichts.
    try:
        conn.execute(
            "INSERT OR IGNORE INTO team_mitgliedschaften (inhaber_id, mitglied_id) "
            "SELECT abo_owner_id, id FROM users WHERE abo_owner_id IS NOT NULL")
        conn.execute(
            "UPDATE users SET aktiver_topf = abo_owner_id "
            "WHERE abo_owner_id IS NOT NULL AND aktiver_topf IS NULL")
        conn.execute("UPDATE users SET abo_owner_id = NULL WHERE abo_owner_id IS NOT NULL")
        conn.execute("UPDATE users SET plan = 'single' WHERE plan = 'pro'")
        conn.execute("UPDATE users SET plan = 'team' WHERE plan = 'lizenz'")
    except Exception as e:
        print(f"Migration warning (abomodell 2026-08): {e}")

    # Gastzugang-Rollen (10.07.2026): bestehende Pruefstatus einmalig in die
    # Pro-Rolle-Tabelle uebernehmen — bisherige Gaeste entsprechen der Rolle
    # 'kunde'. Laeuft NUR, solange image_reviews komplett leer ist (echter
    # Einmal-Backfill, ueberschreibt nie spaetere Rollen-Eintraege).
    try:
        if not conn.execute("SELECT 1 FROM image_reviews LIMIT 1").fetchone():
            conn.execute(
                "INSERT INTO image_reviews (image_id, role, status, reviewed_at) "
                "SELECT id, 'kunde', review_status, reviewed_at FROM images "
                "WHERE review_status IS NOT NULL AND review_status NOT IN ('', 'offen')"
            )
    except Exception as e:
        print(f"Migration warning (image_reviews backfill): {e}")

    # Dashboard (02.06.2026): bestehende Projekte bekommen ihren Dateinamen
    # als Anzeigenamen. Idempotent - setzt nur noch leere Namen.
    try:
        conn.execute("UPDATE projects SET name = filename WHERE name IS NULL OR name = ''")
    except Exception as e:
        print(f"Migration warning (projects.name backfill): {e}")

    # Service-Trennung (02.06.2026): Werkzeug aus project_type ableiten.
    # Idempotent - nur Projekte, die noch das Sammel-Werkzeug 'alttext' tragen.
    try:
        conn.execute("UPDATE projects SET tool='pdf' WHERE tool='alttext' AND project_type='pdf'")
        conn.execute("UPDATE projects SET tool='web' WHERE tool='alttext' AND project_type='url'")
        conn.execute("UPDATE projects SET tool='grafik' WHERE tool='alttext' AND project_type='images'")
    except Exception as e:
        print(f"Migration warning (tool backfill): {e}")

    # Multi-Datei Phase 1 (08.06.2026): Bestehende PDF-Projekte bekommen ihr bisheriges
    # PDF als Dokument 1. Idempotent: nur PDF-Projekte ohne documents-Eintrag.
    # Hintergrund: Vor diesem Branch hatte jedes PDF-Projekt genau eine Datei; die
    # Dokument-Ebene macht das pro Projekt mehrfach moeglich. Damit Single-PDF-Projekte
    # unveraendert funktionieren, schlagen wir hier den Altbestand auf Dokument 1 um.
    #
    # Bugfix 08.06.2026 (Michael-Befund auf Staging): Leere Projekte (kein echtes
    # PDF hochgeladen) haben im `filename`-Feld nach der name-Backfill den
    # Projektnamen stehen — der Backfill wuerde sie sonst falsch als Dokument 1
    # mit 0 Bildern anlegen ("Phantom-Dokument"). Sobald der Nutzer dann seine
    # erste echte PDF anhaengt, landet sie als Dokument 2 — falsch. Fix: nur
    # Projekte, die TATSAECHLICH Bilder haben, bekommen ein Dokument.
    try:
        rows = conn.execute(
            """SELECT id, filename, original_path, extraction_method, total_images
               FROM projects
               WHERE (project_type = 'pdf' OR tool = 'pdf')
                 AND id NOT IN (SELECT project_id FROM documents)
                 AND filename IS NOT NULL AND filename != ''
                 AND EXISTS (SELECT 1 FROM images i WHERE i.project_id = projects.id)"""
        ).fetchall()
        for r in rows:
            cur = conn.execute(
                """INSERT INTO documents
                   (project_id, doc_index, original_filename, original_path,
                    extraction_method, total_images)
                   VALUES (?, 1, ?, ?, ?, ?)""",
                (r["id"], r["filename"], r["original_path"] or "",
                 r["extraction_method"] or "fitz", r["total_images"] or 0)
            )
            doc_id = cur.lastrowid
            conn.execute(
                "UPDATE images SET document_id = ? WHERE project_id = ? AND document_id IS NULL",
                (doc_id, r["id"])
            )
    except Exception as e:
        print(f"Migration warning (documents backfill): {e}")

    # Multi-Datei Phase 2 (14.08.2026): Bestehende WEB-Projekte bekommen ihre
    # bisher flachen Bilder als EINE "Webseite 1" zugeschlagen. Der Altbestand
    # laesst sich nicht mehr pro Einzel-Adresse trennen (die Quell-URL wurde
    # pro Bild nie gespeichert) — deshalb bewusst ein Sammel-Dokument mit der
    # ersten Projekt-Adresse als Namen. Neue Scans legen ab jetzt pro Adresse
    # ein eigenes Dokument an (main.py scan_url). Idempotent wie der
    # PDF-Backfill: nur Web-Projekte ohne documents-Eintrag und mit Bildern.
    try:
        rows = conn.execute(
            """SELECT id, filename, source_url FROM projects
               WHERE (project_type = 'url' OR tool = 'web')
                 AND id NOT IN (SELECT project_id FROM documents)
                 AND EXISTS (SELECT 1 FROM images i WHERE i.project_id = projects.id)"""
        ).fetchall()
        for r in rows:
            quelle = (r["source_url"] or "").strip()
            name = quelle or (r["filename"] or "Webseite")
            cnt = conn.execute(
                "SELECT COUNT(*) FROM images WHERE project_id = ?", (r["id"],)
            ).fetchone()[0]
            cur = conn.execute(
                """INSERT INTO documents
                   (project_id, doc_index, original_filename, original_path,
                    extraction_method, total_images, source_url)
                   VALUES (?, 1, ?, '', 'web', ?, ?)""",
                (r["id"], name[:200], cnt, quelle)
            )
            doc_id = cur.lastrowid
            conn.execute(
                "UPDATE images SET document_id = ? WHERE project_id = ? AND document_id IS NULL",
                (doc_id, r["id"])
            )
        if rows:
            print(f"Migration: Web-Projekte auf Dokument-Ebene umgestellt ({len(rows)} Projekte)")
    except Exception as e:
        print(f"Migration warning (web documents backfill): {e}")

    # Phantom-Dokument-Cleanup (08.06.2026 — Michael-Befund auf Staging):
    # Entfernt Dokument-Zeilen, die KEINEM einzigen Bild zugeordnet sind. Solche
    # Zeilen sind aus dem fehlerhaften ersten Backfill entstanden, als leere
    # PDF-Projekte (kein echter Upload) faelschlich ein "Dokument 1 mit 0 Bildern"
    # bekommen haben — das hat die echte erste PDF, die der Nutzer anschliessend
    # hochlud, auf doc_index 2 verschoben (siehe Michael, PROJ 93).
    #
    # Idempotent: ohne Phantom-Treffer passiert nichts. Nach dem Loeschen werden
    # die verbliebenen Dokumente pro betroffenem Projekt lueckenlos ab 1 in der
    # bisherigen Reihenfolge (doc_index, id) neu nummeriert, damit die echte
    # erste PDF wieder Dokument 1 wird. Keine UNIQUE-Constraint auf
    # (project_id, doc_index) — Renumber kann direkt aufsteigend laufen.
    try:
        phantom_rows = conn.execute(
            """SELECT d.id, d.project_id FROM documents d
               WHERE NOT EXISTS (SELECT 1 FROM images i WHERE i.document_id = d.id)"""
        ).fetchall()
        if phantom_rows:
            phantom_ids = [r["id"] for r in phantom_rows]
            projects_to_renumber = {r["project_id"] for r in phantom_rows}
            placeholders = ",".join("?" * len(phantom_ids))
            conn.execute(
                f"DELETE FROM documents WHERE id IN ({placeholders})",
                phantom_ids,
            )
            for pid in projects_to_renumber:
                remaining = conn.execute(
                    "SELECT id FROM documents WHERE project_id = ? ORDER BY doc_index, id",
                    (pid,),
                ).fetchall()
                for new_idx, row in enumerate(remaining, start=1):
                    conn.execute(
                        "UPDATE documents SET doc_index = ? WHERE id = ?",
                        (new_idx, row["id"]),
                    )
            print(f"Migration: Phantom-Dokumente entfernt ({len(phantom_ids)} Zeilen, {len(projects_to_renumber)} Projekte renumberiert)")
    except Exception as e:
        print(f"Migration warning (phantom document cleanup): {e}")

    conn.commit()


def create_user(email: str, password: str, display_name: str, is_admin: int = 0, email_verified: int = 1) -> int:
    conn = get_db()
    password_hash = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
    try:
        cursor = conn.execute(
            "INSERT INTO users (email, password_hash, display_name, is_admin, email_verified) VALUES (?, ?, ?, ?, ?)",
            (email.lower().strip(), password_hash, display_name.strip(), is_admin, email_verified)
        )
        conn.commit()
        user_id = cursor.lastrowid
    finally:
        conn.close()
    return user_id


def verify_user(email: str, password: str):
    conn = get_db()
    row = conn.execute(
        "SELECT * FROM users WHERE email = ? AND is_active = 1", (email.lower().strip(),)
    ).fetchone()
    conn.close()
    if row and bcrypt.checkpw(password.encode(), row["password_hash"].encode()):
        return dict(row)
    return None


def get_user_by_email(email: str):
    conn = get_db()
    row = conn.execute("SELECT * FROM users WHERE email = ?", (email.lower().strip(),)).fetchone()
    conn.close()
    return dict(row) if row else None


def get_user_by_id(user_id: int):
    conn = get_db()
    row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


def create_password_reset_token(user_id: int) -> str:
    conn = get_db()
    token = secrets.token_urlsafe(32)
    expires = (datetime.utcnow() + timedelta(hours=1)).isoformat()
    # Invalidate old tokens
    conn.execute("UPDATE password_resets SET used = 1 WHERE user_id = ?", (user_id,))
    conn.execute(
        "INSERT INTO password_resets (user_id, token, expires_at) VALUES (?, ?, ?)",
        (user_id, token, expires)
    )
    conn.commit()
    conn.close()
    return token


def verify_reset_token(token: str):
    conn = get_db()
    row = conn.execute(
        "SELECT * FROM password_resets WHERE token = ? AND used = 0", (token,)
    ).fetchone()
    conn.close()
    if not row:
        return None
    if datetime.fromisoformat(row["expires_at"]) < datetime.utcnow():
        return None
    return dict(row)


def reset_password(token: str, new_password: str) -> bool:
    reset = verify_reset_token(token)
    if not reset:
        return False
    conn = get_db()
    password_hash = bcrypt.hashpw(new_password.encode(), bcrypt.gensalt()).decode()
    conn.execute("UPDATE users SET password_hash = ? WHERE id = ?", (password_hash, reset["user_id"]))
    conn.execute("UPDATE password_resets SET used = 1 WHERE id = ?", (reset["id"],))
    conn.commit()
    conn.close()
    return True


def list_all_users():
    """Nutzerliste fuer die Admin-Verwaltung.

    Seit 08.08.2026 mit den Abo-Eckdaten: Die Liste soll auf einen Blick
    zeigen, was jemand gebucht hat — vorher musste man dafuer jede E-Mail
    einzeln in ein anderes Formular tippen. Bewusst eine feste Spaltenliste
    statt SELECT *: So kann kein neues Feld (Passwort-Hash, Stripe-Kennung,
    Token) versehentlich im Browser landen.
    """
    conn = get_db()
    rows = conn.execute(
        "SELECT id, email, display_name, is_admin, admin_level, is_active, "
        "       created_at, last_login, plan, plan_gueltig_bis, "
        "       plan_laufzeit_monate, plan_quelle, auto_verlaengerung, "
        "       team_name, abo_owner_id, geplanter_plan, api_tageslimit "
        "FROM users ORDER BY created_at DESC"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def update_user_active(user_id: int, is_active: int):
    conn = get_db()
    conn.execute("UPDATE users SET is_active = ? WHERE id = ?", (is_active, user_id))
    conn.commit()
    conn.close()


def set_user_language(user_id: int, language: str):
    """Speichert die UI-Sprach-Praeferenz eines Users (ISO-639-1: de/en/fr/es).

    Wird beim bewussten Sprachwechsel eines eingeloggten Nutzers aufgerufen
    (siehe set-language-Route). Die Alt-Text-Ausgabesprache bleibt davon
    unberuehrt - das ist ein separates Feature.
    """
    conn = get_db()
    conn.execute("UPDATE users SET language = ? WHERE id = ?", (language, user_id))
    conn.commit()
    conn.close()


def delete_user_data(user_id: int):
    """Delete all projects and images for a user (DSGVO)."""
    conn = get_db()
    projects = conn.execute("SELECT id FROM projects WHERE user_id = ?", (user_id,)).fetchall()
    for p in projects:
        # Selbstloeschung (08.08.2026): SQLite erzwingt Fremdschluessel nur mit
        # PRAGMA foreign_keys=ON — auf das ON DELETE CASCADE im Schema ist also
        # kein Verlass. Alles, was an einem Projekt haengt, wird ausdruecklich
        # mitgeloescht, sonst bleiben Alt-Texte, Chatverlaeufe und Gast-Links
        # (mit gueltigem Token!) als Waisen zurueck.
        conn.execute("DELETE FROM image_reviews WHERE image_id IN "
                     "(SELECT id FROM images WHERE project_id = ?)", (p["id"],))
        conn.execute("DELETE FROM images WHERE project_id = ?", (p["id"],))
        # Multi-Datei (08.06.2026): Dokumente eines Projekts mit aufraeumen.
        conn.execute("DELETE FROM documents WHERE project_id = ?", (p["id"],))
        conn.execute("DELETE FROM chat_messages WHERE project_id = ?", (p["id"],))
        conn.execute("DELETE FROM messages WHERE project_id = ?", (p["id"],))
        conn.execute("DELETE FROM shares WHERE project_id = ?", (p["id"],))
    conn.execute("DELETE FROM projects WHERE user_id = ?", (user_id,))
    conn.execute("DELETE FROM shares WHERE created_by = ?", (user_id,))
    conn.execute("DELETE FROM messages WHERE sender_user_id = ? OR recipient_user_id = ?",
                 (user_id, user_id))
    conn.execute("DELETE FROM password_resets WHERE user_id = ?", (user_id,))
    # Personenbezogene Nebenspuren (DSGVO): Nutzung, API-Ergebnisse, offene
    # Adress-Wechsel und Bestaetigungen, eigene Prompt-Vorlagen.
    conn.execute("DELETE FROM api_usage WHERE user_id = ?", (user_id,))
    conn.execute("DELETE FROM api_results WHERE user_id = ?", (user_id,))
    conn.execute("DELETE FROM email_changes WHERE user_id = ?", (user_id,))
    conn.execute("DELETE FROM email_verifications WHERE user_id = ?", (user_id,))
    conn.execute("DELETE FROM user_prompts WHERE user_id = ?", (user_id,))
    # Abrechnungsspuren: sowohl die eigenen Buchungen als auch die, die in
    # SEINEN Topf gelaufen sind (Team-Inhaber) — sonst rechnet billing spaeter
    # gegen ein Konto, das es nicht mehr gibt.
    conn.execute("DELETE FROM usage_events WHERE user_id = ? OR konto_user_id = ?",
                 (user_id, user_id))
    conn.execute("DELETE FROM paket_abbuchungen WHERE konto_user_id = ?", (user_id,))
    conn.execute("DELETE FROM quota_pakete WHERE user_id = ?", (user_id,))
    # Review-Befund 4 (31.07.2026) / Abomodell 06.08.2026: Wird ein TEAM-
    # INHABER geloescht, duerfen keine baumelnden Mitgliedschaften bleiben —
    # billing wuerde sonst gegen ein nicht existentes Konto rechnen. Beide
    # Richtungen aufraeumen (als Inhaber UND als Mitglied); wer gerade in
    # diesem Topf arbeitete, faellt auf den eigenen Topf zurueck. Offene
    # Einladungen des Geloeschten werden mit entsorgt, damit kein Token mehr
    # auf ein totes Konto zeigen kann.
    conn.execute("DELETE FROM team_mitgliedschaften WHERE inhaber_id = ? OR mitglied_id = ?",
                 (user_id, user_id))
    conn.execute("UPDATE users SET aktiver_topf = NULL WHERE aktiver_topf = ?", (user_id,))
    conn.execute("UPDATE users SET abo_owner_id = NULL WHERE abo_owner_id = ?", (user_id,))
    conn.execute("DELETE FROM team_einladungen WHERE inhaber_id = ?", (user_id,))
    # Auch Einladungen AN diese Adresse: sie zeigen sonst weiter auf eine
    # E-Mail, deren Konto nicht mehr existiert.
    conn.execute("DELETE FROM team_einladungen WHERE lower(email) = "
                 "(SELECT lower(email) FROM users WHERE id = ?)", (user_id,))
    # Altmodell Lizenzschluessel: Verweis loesen statt Schluessel loeschen
    # (er kann noch bei einem Kunden liegen und muss sichtbar bleiben).
    conn.execute("UPDATE lizenzschluessel SET inhaber_user_id = NULL "
                 "WHERE inhaber_user_id = ?", (user_id,))
    conn.execute("UPDATE lizenzschluessel SET erstellt_von = NULL WHERE erstellt_von = ?",
                 (user_id,))
    conn.execute("DELETE FROM api_keys WHERE user_id = ?", (user_id,))
    conn.execute("DELETE FROM users WHERE id = ?", (user_id,))
    conn.commit()
    conn.close()


def admin_reset_password(user_id: int, new_password: str):
    conn = get_db()
    password_hash = bcrypt.hashpw(new_password.encode(), bcrypt.gensalt()).decode()
    conn.execute("UPDATE users SET password_hash = ? WHERE id = ?", (password_hash, user_id))
    conn.commit()
    conn.close()


# ─── Administrator-Verwaltung (14.06.2026) ───────────────────────
# Zwei Rechte-Stufen pro Admin (Spalte users.admin_level):
#   'full' = Voll-Admin – alle Rechte wie der Gruender-Account
#            (Nutzer sperren, loeschen, Passwoerter zuruecksetzen,
#             andere zu Admins machen).
#   'view' = Nur-Einsicht – sieht die Nutzerliste und wer sich
#            angemeldet hat, darf aber nichts veraendern.

def list_admins():
    """Alle Administratoren (is_admin=1) inkl. ihrer Rechte-Stufe."""
    conn = get_db()
    rows = conn.execute(
        "SELECT id, email, display_name, admin_level, is_active "
        "FROM users WHERE is_admin = 1 ORDER BY created_at"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def count_full_admins():
    """Anzahl der Voll-Admins – Schutz gegen versehentliches Aussperren."""
    conn = get_db()
    n = conn.execute(
        "SELECT COUNT(*) FROM users WHERE is_admin = 1 AND admin_level = 'full'"
    ).fetchone()[0]
    conn.close()
    return n


def set_user_admin(user_id: int, is_admin: int, admin_level: str = "full"):
    """Admin-Status und Rechte-Stufe eines Nutzers setzen.

    is_admin=0 entfernt die Admin-Rechte (das Konto bleibt als normaler
    Nutzer bestehen). admin_level wird nur ausgewertet, wenn is_admin=1
    und ist 'full' oder 'view'.
    """
    if admin_level not in ("full", "view"):
        admin_level = "full"
    conn = get_db()
    conn.execute(
        "UPDATE users SET is_admin = ?, admin_level = ? WHERE id = ?",
        (1 if is_admin else 0, admin_level, user_id),
    )
    conn.commit()
    conn.close()


# ─── Email Change Verification ────────────────────────────────

def create_email_change_token(user_id: int, new_email: str) -> str:
    """Create a token for email change verification. Returns the raw token."""
    conn = get_db()
    token = secrets.token_urlsafe(32)
    expires = (datetime.utcnow() + timedelta(hours=1)).isoformat()
    # Invalidate old pending changes for this user
    conn.execute("UPDATE email_changes SET used = 1 WHERE user_id = ? AND used = 0", (user_id,))
    conn.execute(
        "INSERT INTO email_changes (user_id, new_email, token, expires_at) VALUES (?, ?, ?, ?)",
        (user_id, new_email, token, expires)
    )
    conn.commit()
    conn.close()
    return token


def verify_email_change_token(token: str) -> dict | None:
    """Verify an email change token. Returns dict with user_id and new_email, or None."""
    conn = get_db()
    row = conn.execute(
        "SELECT * FROM email_changes WHERE token = ? AND used = 0", (token,)
    ).fetchone()
    conn.close()
    if not row:
        return None
    if datetime.fromisoformat(row["expires_at"]) < datetime.utcnow():
        return None
    return dict(row)


def confirm_email_change(token: str) -> dict | None:
    """Confirm the email change: update user email and mark token as used.
    Returns dict with user_id and new_email on success, None on failure."""
    change = verify_email_change_token(token)
    if not change:
        return None
    conn = get_db()
    # Check new email is still available
    existing = conn.execute("SELECT id FROM users WHERE email = ?", (change["new_email"],)).fetchone()
    if existing:
        conn.close()
        return None
    conn.execute("UPDATE users SET email = ? WHERE id = ?", (change["new_email"], change["user_id"]))
    conn.execute("UPDATE email_changes SET used = 1 WHERE id = ?", (change["id"],))
    conn.commit()
    conn.close()
    return {"user_id": change["user_id"], "new_email": change["new_email"]}




# ─── Daily Usage Limits ──────────────────────────────────────

def get_daily_image_count(user_id: int) -> int:
    """Count how many images a user has processed today (UTC)."""
    conn = get_db()
    row = conn.execute(
        """SELECT COUNT(*) FROM images i
           JOIN projects p ON i.project_id = p.id
           WHERE p.user_id = ? AND i.status IN ('done', 'processing')
           AND date(i.created_at) = date('now')""",
        (user_id,)
    ).fetchone()
    conn.close()
    return row[0] if row else 0


def get_daily_api_count(user_id: int) -> int:
    """Count how many API calls a user has made today (UTC)."""
    conn = get_db()
    row = conn.execute(
        """SELECT COUNT(*) FROM api_usage
           WHERE user_id = ? AND date(timestamp) = date('now')""",
        (user_id,)
    ).fetchone()
    conn.close()
    return row[0] if row else 0


# ─── Email Verification (Registration) ───────────────────────

def create_email_verification_token(user_id: int, email: str) -> str:
    """Create a token for registration email verification. Returns the raw token."""
    conn = get_db()
    token = secrets.token_urlsafe(32)
    expires = (datetime.utcnow() + timedelta(hours=24)).isoformat()
    conn.execute("UPDATE email_verifications SET used = 1 WHERE user_id = ? AND used = 0", (user_id,))
    conn.execute(
        "INSERT INTO email_verifications (user_id, email, token, expires_at) VALUES (?, ?, ?, ?)",
        (user_id, email, token, expires)
    )
    conn.commit()
    conn.close()
    return token


def verify_email_verification_token(token: str):
    """Verify a registration email token. Returns dict or None."""
    conn = get_db()
    row = conn.execute(
        "SELECT * FROM email_verifications WHERE token = ? AND used = 0", (token,)
    ).fetchone()
    conn.close()
    if not row:
        return None
    if datetime.fromisoformat(row["expires_at"]) < datetime.utcnow():
        return None
    return dict(row)


def mark_email_verified(token: str):
    """Confirm email verification: set email_verified=1 and mark token used.
    Returns dict with user_id and email on success, None on failure."""
    verification = verify_email_verification_token(token)
    if not verification:
        return None
    conn = get_db()
    conn.execute("UPDATE users SET email_verified = 1 WHERE id = ?", (verification["user_id"],))
    conn.execute("UPDATE email_verifications SET used = 1 WHERE id = ?", (verification["id"],))
    conn.commit()
    conn.close()
    return {"user_id": verification["user_id"], "email": verification["email"]}


def resend_verification_token(email: str):
    """Create a new verification token for an unverified user. Returns token or None."""
    conn = get_db()
    row = conn.execute(
        "SELECT id, email_verified FROM users WHERE email = ? AND is_active = 1", (email.lower().strip(),)
    ).fetchone()
    conn.close()
    if not row:
        return None
    # Already verified
    try:
        if row["email_verified"]:
            return None
    except (IndexError, KeyError):
        return None
    return create_email_verification_token(row["id"], email.lower().strip())


# ─── API Key Management ──────────────────────────────────────

def _hash_api_key(key: str) -> str:
    """SHA-256 hash of an API key for storage."""
    return hashlib.sha256(key.encode()).hexdigest()


def create_api_key(user_id: int, name: str) -> tuple[int, str]:
    """Create a new API key for a user.

    Returns:
        Tuple of (key_id, raw_key). The raw key is only returned once.
    """
    raw_key = f"idocs_{secrets.token_urlsafe(32)}"
    key_hash = _hash_api_key(raw_key)
    conn = get_db()
    try:
        cursor = conn.execute(
            "INSERT INTO api_keys (user_id, key_hash, name) VALUES (?, ?, ?)",
            (user_id, key_hash, name.strip())
        )
        conn.commit()
        key_id = cursor.lastrowid
    finally:
        conn.close()
    return key_id, raw_key


def verify_api_key(raw_key: str) -> dict | None:
    """Verify an API key and return the associated user info.

    Also updates last_used timestamp.

    Returns:
        Dict with id, user_id, name, or None if invalid/inactive.
    """
    key_hash = _hash_api_key(raw_key)
    conn = get_db()
    row = conn.execute(
        """SELECT ak.id, ak.user_id, ak.name, u.email, u.is_active as user_active
           FROM api_keys ak
           JOIN users u ON ak.user_id = u.id
           WHERE ak.key_hash = ? AND ak.is_active = 1""",
        (key_hash,)
    ).fetchone()
    if not row:
        conn.close()
        return None
    if not row["user_active"]:
        conn.close()
        return None
    # Update last_used
    conn.execute(
        "UPDATE api_keys SET last_used = datetime('now') WHERE id = ?",
        (row["id"],)
    )
    conn.commit()
    conn.close()
    return {
        "id": row["id"],
        "user_id": row["user_id"],
        "name": row["name"],
        "email": row["email"],
    }


def list_api_keys(user_id: int) -> list[dict]:
    """List all API keys for a user (without the actual key hash)."""
    conn = get_db()
    rows = conn.execute(
        "SELECT id, name, created_at, last_used, is_active FROM api_keys WHERE user_id = ? ORDER BY created_at DESC",
        (user_id,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def delete_api_key(user_id: int, key_id: int) -> bool:
    """Delete an API key. Returns True if deleted, False if not found."""
    conn = get_db()
    cursor = conn.execute(
        "DELETE FROM api_keys WHERE id = ? AND user_id = ?",
        (key_id, user_id)
    )
    conn.commit()
    deleted = cursor.rowcount > 0
    conn.close()
    return deleted


def rename_api_key(user_id: int, key_id: int, new_name: str) -> bool:
    """Rename an API key. Returns True if successful."""
    new_name = new_name.strip()
    if not new_name:
        return False
    conn = get_db()
    try:
        cursor = conn.execute(
            "UPDATE api_keys SET name = ? WHERE id = ? AND user_id = ?",
            (new_name, key_id, user_id)
        )
        conn.commit()
        return cursor.rowcount > 0
    finally:
        conn.close()


# ─── API Usage Tracking ──────────────────────────────────────

def log_api_usage(api_key_id: int, user_id: int, processing_time_ms: int = 0,
                  model_used: str = "", image_size_bytes: int = 0,
                  success: bool = True, error_message: str = ""):
    """Log a single API call for usage tracking."""
    conn = get_db()
    conn.execute(
        """INSERT INTO api_usage (api_key_id, user_id, processing_time_ms, model_used,
           image_size_bytes, success, error_message)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (api_key_id, user_id, processing_time_ms, model_used, image_size_bytes,
         1 if success else 0, error_message)
    )
    conn.commit()
    conn.close()


def get_api_usage_stats(user_id: int) -> dict:
    """Get usage statistics for a user across all their API keys."""
    conn = get_db()

    total = conn.execute(
        "SELECT COUNT(*) FROM api_usage WHERE user_id = ?", (user_id,)
    ).fetchone()[0]

    week = conn.execute(
        """SELECT COUNT(*) FROM api_usage
           WHERE user_id = ? AND timestamp >= date('now', 'weekday 1', '-7 days')""",
        (user_id,)
    ).fetchone()[0]

    month = conn.execute(
        """SELECT COUNT(*) FROM api_usage
           WHERE user_id = ? AND timestamp >= date('now', 'start of month')""",
        (user_id,)
    ).fetchone()[0]

    last_row = conn.execute(
        "SELECT timestamp FROM api_usage WHERE user_id = ? ORDER BY timestamp DESC LIMIT 1",
        (user_id,)
    ).fetchone()
    last_call = last_row[0] if last_row else None

    success_count = conn.execute(
        "SELECT COUNT(*) FROM api_usage WHERE user_id = ? AND success = 1", (user_id,)
    ).fetchone()[0]

    conn.close()
    return {
        "total_calls": total,
        "calls_this_week": week,
        "calls_this_month": month,
        "last_call": last_call,
        "success_rate": round(success_count / total * 100, 1) if total > 0 else 0,
    }


def create_api_result(result_id: str, user_id: int, api_key_id: int,
                      alt_text: str, langbeschreibung: str, bildtyp: str = "unbekannt",
                      konfidenz: str = "mittel", model_used: str = "",
                      processing_time_ms: int = 0, language: str = "de",
                      context_text: str = "") -> str:
    """Speichert ein API-Ergebnis und gibt die result_id zurueck."""
    conn = get_db()
    try:
        conn.execute(
            """INSERT INTO api_results
               (id, user_id, api_key_id, alt_text, langbeschreibung, bildtyp,
                konfidenz, model_used, processing_time_ms, language, context_text)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (result_id, user_id, api_key_id, alt_text, langbeschreibung, bildtyp,
             konfidenz, model_used, processing_time_ms, language, context_text)
        )
        conn.commit()
    finally:
        conn.close()
    return result_id


def get_api_result(result_id: str, user_id: int):
    """Holt ein API-Ergebnis. Prueft dass es dem User gehoert."""
    conn = get_db()
    row = conn.execute(
        "SELECT * FROM api_results WHERE id = ? AND user_id = ?",
        (result_id, user_id)
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def update_api_result(result_id: str, user_id: int,
                      alt_text=None,
                      langbeschreibung=None) -> bool:
    """Aktualisiert Alt-Text und/oder Langbeschreibung eines API-Ergebnisses."""
    conn = get_db()
    row = conn.execute(
        "SELECT id FROM api_results WHERE id = ? AND user_id = ?",
        (result_id, user_id)
    ).fetchone()
    if not row:
        conn.close()
        return False

    updates = []
    params = []
    if alt_text is not None:
        updates.append("alt_text = ?")
        params.append(alt_text)
    if langbeschreibung is not None:
        updates.append("langbeschreibung = ?")
        params.append(langbeschreibung)

    if not updates:
        conn.close()
        return True  # Nichts zu aendern

    updates.append("updated_at = datetime('now')")
    params.extend([result_id, user_id])

    conn.execute(
        f"UPDATE api_results SET {', '.join(updates)} WHERE id = ? AND user_id = ?",
        params
    )
    conn.commit()
    conn.close()
    return True
