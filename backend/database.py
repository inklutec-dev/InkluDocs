import sqlite3
import os
import bcrypt
import secrets
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

        CREATE INDEX IF NOT EXISTS idx_projects_user ON projects(user_id);
        CREATE INDEX IF NOT EXISTS idx_images_project ON images(project_id);
    """)
    conn.commit()
    conn.close()


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
    conn.execute("DELETE FROM users WHERE id = ?", (user_id,))
    conn.commit()
    conn.close()


def admin_reset_password(user_id: int, new_password: str):
    conn = get_db()
    password_hash = bcrypt.hashpw(new_password.encode(), bcrypt.gensalt()).decode()
    conn.execute("UPDATE users SET password_hash = ? WHERE id = ?", (password_hash, user_id))
    conn.commit()
    conn.close()
