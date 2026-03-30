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
    """)
    conn.commit()

    # Backward-compatible migrations using ALTER TABLE with try/except
    _migrate_columns(conn)

    conn.close()


def _migrate_columns(conn):
    """Add new columns to existing tables. Uses try/except for idempotency."""
    migrations = [
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
        ("images", "langbeschreibung", "ALTER TABLE images ADD COLUMN langbeschreibung TEXT DEFAULT ''"),
        ("images", "original_alt", "ALTER TABLE images ADD COLUMN original_alt TEXT DEFAULT ''"),
        ("images", "feedback", "ALTER TABLE images ADD COLUMN feedback TEXT DEFAULT ''"),
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

    conn.commit()


def create_user(email: str, password: str, display_name: str, is_admin: int = 0) -> int:
    conn = get_db()
    password_hash = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
    try:
        cursor = conn.execute(
            "INSERT INTO users (email, password_hash, display_name, is_admin) VALUES (?, ?, ?, ?)",
            (email.lower().strip(), password_hash, display_name.strip(), is_admin)
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
    conn = get_db()
    rows = conn.execute(
        "SELECT id, email, display_name, is_admin, is_active, created_at, last_login FROM users ORDER BY created_at DESC"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def update_user_active(user_id: int, is_active: int):
    conn = get_db()
    conn.execute("UPDATE users SET is_active = ? WHERE id = ?", (is_active, user_id))
    conn.commit()
    conn.close()


def delete_user_data(user_id: int):
    """Delete all projects and images for a user (DSGVO)."""
    conn = get_db()
    projects = conn.execute("SELECT id FROM projects WHERE user_id = ?", (user_id,)).fetchall()
    for p in projects:
        conn.execute("DELETE FROM images WHERE project_id = ?", (p["id"],))
    conn.execute("DELETE FROM projects WHERE user_id = ?", (user_id,))
    conn.execute("DELETE FROM password_resets WHERE user_id = ?", (user_id,))
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
