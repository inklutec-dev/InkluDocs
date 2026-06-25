import os
import re
import shutil
import json
import csv
import asyncio
import secrets
import time
import io
from collections import defaultdict
from datetime import datetime, timedelta
from contextlib import asynccontextmanager
from urllib.parse import urljoin, urlparse
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

import httpx
from bs4 import BeautifulSoup

import demo as demo_mod  # Demo-Modus (oeffentliche Kostprobe ohne Anmeldung) — nur aktiv bei DEMO_MODE=on
from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, Request, Form
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from jose import jwt, JWTError

from database import (
    init_db, get_db, create_user, verify_user, get_user_by_email, get_user_by_id,
    create_password_reset_token, verify_reset_token, reset_password,
    list_all_users, update_user_active, delete_user_data, admin_reset_password,
    list_admins, count_full_admins, set_user_admin,
    create_api_key, verify_api_key, list_api_keys, delete_api_key, rename_api_key,
    log_api_usage, get_api_usage_stats,
    create_email_change_token, confirm_email_change,
    create_api_result, get_api_result, update_api_result,
    create_email_verification_token, mark_email_verified, resend_verification_token,
    get_daily_image_count, get_daily_api_count,
)
from pdf_processor import extract_images_from_pdf, generate_alt_text, generate_alt_text_for_image, clear_project_cache
import sharing  # Gastzugang / Projekt-Freigabe (19.06.2026)
from i18n import get_templates, detect_language, template_context, SUPPORTED_LANGUAGES
from tools import TOOLS, is_valid_tool_key

# Generate a persistent SECRET_KEY if not set
SECRET_KEY_FILE = "/app/data/.secret_key"
def _get_secret_key():
    env_key = os.environ.get("SECRET_KEY", "")
    if env_key and env_key != "inkludocs-production-key-2025":
        return env_key
    if os.path.exists(SECRET_KEY_FILE):
        return open(SECRET_KEY_FILE).read().strip()
    key = secrets.token_hex(32)
    os.makedirs(os.path.dirname(SECRET_KEY_FILE), exist_ok=True)
    with open(SECRET_KEY_FILE, "w") as f:
        f.write(key)
    return key

SECRET_KEY = _get_secret_key()
ALGORITHM = "HS256"
TOKEN_EXPIRE_HOURS = 24
UPLOAD_DIR = "/app/data/uploads"
RESULTS_DIR = "/app/data/results"
BASE_URL = os.environ.get("BASE_URL", "https://inkludocs.inklutec.de")
MAX_UPLOAD_SIZE = 50 * 1024 * 1024  # 50 MB

# SMTP configuration for email sending
SMTP_SERVER = os.environ.get("SMTP_SERVER", "w01ccfc3.kasserver.com")
SMTP_PORT = int(os.environ.get("SMTP_PORT", "587"))
SMTP_USER = os.environ.get("SMTP_USER", "kontakt@inklutec.de")
SMTP_PASS = os.environ.get("SMTP_PASS", "")
SMTP_FROM = os.environ.get("SMTP_FROM", "kontakt@inklutec.de")
NOTIFICATION_EMAIL = os.environ.get("NOTIFICATION_EMAIL", SMTP_FROM)
DAILY_IMAGE_LIMIT = int(os.environ.get("DAILY_IMAGE_LIMIT", "100"))


def send_email(to_email: str, subject: str, html_body: str, bcc_admin: bool = True, attachment_path: str = None, reply_to: str = None, from_name: str = None) -> bool:
    """Send an email via SMTP with optional file attachment.

    Args:
        bcc_admin: If True, send BCC copy to admin. Set False for
                   sensitive emails like password resets.
        attachment_path: Optional path to a file to attach (e.g. image).
    """
    if not SMTP_PASS:
        print(f"E-Mail nicht gesendet (kein SMTP-Passwort konfiguriert): {subject} an {to_email}")
        return False
    try:
        from email.mime.base import MIMEBase
        from email import encoders

        msg = MIMEMultipart("mixed")
        if "staging" in BASE_URL:
            subject = f"[STAGING] {subject}"
        msg["Subject"] = subject
        from email.utils import formataddr
        # Nur den Anzeigenamen kodieren, Adresse bleibt gueltig (sonst lehnt Gmail mit 550 5.7.1 ab).
        msg["From"] = formataddr((from_name, SMTP_FROM), charset="utf-8") if from_name else f"InkluDocs <{SMTP_FROM}>"
        if reply_to:
            msg["Reply-To"] = reply_to
        msg["To"] = to_email

        html_part = MIMEMultipart("alternative")
        html_part.attach(MIMEText(html_body, "html", "utf-8"))
        msg.attach(html_part)

        if attachment_path and os.path.exists(attachment_path):
            try:
                with open(attachment_path, "rb") as f:
                    part = MIMEBase("application", "octet-stream")
                    part.set_payload(f.read())
                encoders.encode_base64(part)
                filename = os.path.basename(attachment_path)
                part.add_header("Content-Disposition", f"attachment; filename={filename}")
                msg.attach(part)
            except Exception as e:
                print(f"Anhang konnte nicht hinzugefügt werden: {e}")

        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT, timeout=15)
        server.starttls()
        server.login(SMTP_USER, SMTP_PASS)
        recipients = [to_email]
        if bcc_admin and to_email != NOTIFICATION_EMAIL:
            msg["Bcc"] = NOTIFICATION_EMAIL
            recipients.append(NOTIFICATION_EMAIL)
        server.sendmail(SMTP_FROM, recipients, msg.as_string())
        server.quit()
        print(f"E-Mail gesendet an {to_email}: {subject}")
        return True
    except Exception as e:
        print(f"E-Mail-Fehler ({to_email}): {e}")
        return False


# Allowed image extensions for direct upload
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".heic", ".heif", ".bmp", ".tiff", ".tif"}
# Note: SVG intentionally excluded – PIL/Pillow cannot open SVGs, causes processing crash

# Rate limiting for login
_login_attempts = defaultdict(list)
MAX_LOGIN_ATTEMPTS = 5
LOGIN_WINDOW_SECONDS = 300  # 5 minutes

# Rate limiting for API (per API key)
_api_rate_minute = defaultdict(list)  # key_id -> [timestamps]
_api_rate_day = defaultdict(list)     # key_id -> [timestamps]
API_RATE_LIMIT_MINUTE = 60
API_RATE_LIMIT_DAY = 1000


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    # Clean up stale API temp files from previous runs/crashes
    api_tmp = os.path.join(UPLOAD_DIR, "api_tmp")
    if os.path.exists(api_tmp):
        cutoff = time.time() - 3600
        for f in os.listdir(api_tmp):
            fp = os.path.join(api_tmp, f)
            try:
                if os.path.isfile(fp) and os.path.getmtime(fp) < cutoff:
                    os.remove(fp)
            except Exception:
                pass
    # Create default admin user only if NO users exist at all (fresh install)
    conn = get_db()
    user_count = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
    conn.close()
    if user_count == 0:
        try:
            create_user("kontakt@inklutec.de", "inkludocs2025", "Administrator", is_admin=1)
            print("Default admin user created (fresh install)")
        except Exception:
            pass

    # ── Demo-Modus: periodischer Aufraeumer fuer abgelaufene Sitzungen ──
    # Laeuft NUR bei DEMO_MODE=on (eigene Demo-Instanz). Entfernt Wegwerf-Sitzungen
    # (Nutzer + Projekte + Bilder + Dateien), die laenger als das gleitende TTL
    # untaetig waren. In Production/Staging wird der Task gar nicht erst gestartet.
    if demo_mod.demo_enabled():
        demo_mod.ensure_demo_schema()  # Zaehler-Tabelle der Demo anlegen (idempotent)
        async def _demo_cleanup_loop():
            while True:
                try:
                    n = await asyncio.get_event_loop().run_in_executor(
                        None, demo_mod.cleanup_expired_sessions, UPLOAD_DIR, RESULTS_DIR
                    )
                    if n:
                        print(f"[demo] {n} abgelaufene Sitzung(en) aufgeraeumt")
                except Exception as e:
                    print(f"[demo] Aufraeum-Fehler: {e}")
                await asyncio.sleep(600)  # alle 10 Minuten
        asyncio.create_task(_demo_cleanup_loop())

    yield


app = FastAPI(title="InkluDocs", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://inkludocs.inklutec.de", "https://staging.inkludocs.inklutec.de"],
    allow_methods=["GET", "POST", "PATCH", "DELETE"],
    allow_headers=["Content-Type", "X-API-Key"],
    allow_credentials=True,
    expose_headers=["X-Export-Warnings", "X-Export-Tagged", "X-Export-Total",
                    "X-RateLimit-Remaining-Minute", "X-RateLimit-Remaining-Day",
                    "X-RateLimit-Limit", "X-RateLimit-Remaining", "X-RateLimit-Reset", "Retry-After"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="/app/frontend"), name="static")

# Jinja2-Templates (fuer i18n-Seiten, inkrementell migriert)
templates = get_templates()


def create_token(user_id: int, email: str, is_admin: int = 0) -> str:
    expire = datetime.utcnow() + timedelta(hours=TOKEN_EXPIRE_HOURS)
    return jwt.encode(
        {"sub": str(user_id), "email": email, "is_admin": is_admin, "exp": expire},
        SECRET_KEY, algorithm=ALGORITHM,
    )


def get_current_user(request: Request) -> dict:
    token = request.cookies.get("token")
    if not token:
        raise HTTPException(status_code=401, detail="Nicht angemeldet")
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return {"id": int(payload["sub"]), "email": payload["email"], "is_admin": payload.get("is_admin", 0)}
    except JWTError:
        raise HTTPException(status_code=401, detail="Token ungueltig")


def require_admin(request: Request) -> dict:
    """Jeder Admin (Voll-Admin oder Nur-Einsicht) – lesender Zugriff."""
    user = get_current_user(request)
    if not user.get("is_admin"):
        raise HTTPException(status_code=403, detail="Nur fuer Administratoren")
    return user


def require_full_admin(request: Request) -> dict:
    """Nur Voll-Admins – fuer veraendernde Aktionen (Nutzer sperren, loeschen,
    Passwoerter zuruecksetzen, Konten anlegen, Admins verwalten).

    Liest die aktuelle Rechte-Stufe frisch aus der Datenbank, damit ein
    entzogenes oder herabgestuftes Recht sofort greift – nicht erst nach
    Ablauf des Login-Tokens.
    """
    user = get_current_user(request)
    if not user.get("is_admin"):
        raise HTTPException(status_code=403, detail="Nur fuer Administratoren")
    db_user = get_user_by_id(user["id"])
    if not db_user or not db_user.get("is_admin") or db_user.get("admin_level") != "full":
        raise HTTPException(status_code=403, detail="Nur fuer Voll-Administratoren")
    return user


# --- Gastzugang / Projekt-Freigabe (19.06.2026) ---
# Gast-Sitzung = eigenes guest_token-Cookie, an GENAU EINEN Freigabe-Token
# gebunden. Spiegelt das bestehende JWT-Auth-Muster (create_token / get_current_user),
# kein Eigenbau. Der eingeloggte Pfad bleibt voellig unberuehrt.
def create_guest_token(share_token, guest_email):
    """JWT fuer eine bestaetigte Gast-Sitzung (nach erfolgreichem E-Mail-Gate)."""
    expire = datetime.utcnow() + timedelta(hours=TOKEN_EXPIRE_HOURS)
    return jwt.encode(
        {"type": "guest", "share_token": share_token, "guest": guest_email, "exp": expire},
        SECRET_KEY, algorithm=ALGORITHM,
    )


def get_guest_session(request, share_token):
    """Liest das guest_token-Cookie und liefert die Gast-Sitzung NUR, wenn sie zu
    diesem share_token gehoert. Sonst None (kein Zugriff)."""
    gt = request.cookies.get("guest_token")
    if not gt:
        return None
    try:
        payload = jwt.decode(gt, SECRET_KEY, algorithms=[ALGORITHM])
    except JWTError:
        return None
    if payload.get("type") != "guest" or payload.get("share_token") != share_token:
        return None
    return {"guest": payload.get("guest"), "share_token": share_token}


def get_api_user(request: Request) -> dict:
    """Authenticate via X-API-Key header for public API endpoints."""
    api_key = request.headers.get("X-API-Key", "")
    if not api_key:
        raise HTTPException(status_code=401, detail="API-Key fehlt. Bitte X-API-Key Header setzen.")
    key_info = verify_api_key(api_key)
    if not key_info:
        raise HTTPException(status_code=401, detail="API-Key ungueltig oder deaktiviert.")
    return {"id": key_info["user_id"], "email": key_info["email"], "is_admin": 0, "api_key_id": key_info["id"]}


def check_api_rate_limit(api_key_id: int):
    """Check and enforce rate limits for an API key. Raises 429 if exceeded."""
    now = time.time()

    # Clean old entries and check minute limit
    _api_rate_minute[api_key_id] = [t for t in _api_rate_minute[api_key_id] if now - t < 60]
    if len(_api_rate_minute[api_key_id]) >= API_RATE_LIMIT_MINUTE:
        raise HTTPException(
            status_code=429,
            detail=f"Rate-Limit ueberschritten: max. {API_RATE_LIMIT_MINUTE} Anfragen pro Minute.",
            headers={
                "X-RateLimit-Limit": str(API_RATE_LIMIT_MINUTE),
                "X-RateLimit-Remaining": "0",
                "X-RateLimit-Reset": str(int(min(_api_rate_minute[api_key_id]) + 60)),
                "Retry-After": "60",
            }
        )

    # Clean old entries and check daily limit
    _api_rate_day[api_key_id] = [t for t in _api_rate_day[api_key_id] if now - t < 86400]
    if len(_api_rate_day[api_key_id]) >= API_RATE_LIMIT_DAY:
        raise HTTPException(
            status_code=429,
            detail=f"Tageslimit ueberschritten: max. {API_RATE_LIMIT_DAY} Anfragen pro Tag.",
            headers={
                "X-RateLimit-Limit": str(API_RATE_LIMIT_DAY),
                "X-RateLimit-Remaining": "0",
                "Retry-After": "3600",
            }
        )

    # Record this request
    _api_rate_minute[api_key_id].append(now)
    _api_rate_day[api_key_id].append(now)

    return {
        "minute_remaining": API_RATE_LIMIT_MINUTE - len(_api_rate_minute[api_key_id]),
        "day_remaining": API_RATE_LIMIT_DAY - len(_api_rate_day[api_key_id]),
    }


# ─── Auth Routes ─────────────────────────────────────────────

@app.post("/api/login")
async def login(request: Request):
    # Rate limiting
    client_ip = request.client.host if request.client else "unknown"
    now = time.time()
    _login_attempts[client_ip] = [t for t in _login_attempts[client_ip] if now - t < LOGIN_WINDOW_SECONDS]
    if len(_login_attempts[client_ip]) >= MAX_LOGIN_ATTEMPTS:
        raise HTTPException(status_code=429, detail="Zu viele Anmeldeversuche. Bitte 5 Minuten warten.")

    data = await request.json()
    email = data.get("email", "").strip()
    password = data.get("password", "")
    user = verify_user(email, password)
    if not user:
        _login_attempts[client_ip].append(now)
        raise HTTPException(status_code=401, detail="E-Mail oder Passwort falsch")

    # Check email verification
    if not user.get("email_verified", 1):
        # Steve 08.06.2026: echte Umlaute auch im Backend-Detail-String. Das Frontend
        # erkennt den Fehlerpfad ueber msg.includes('nicht bestätigt') — bleibt damit
        # synchron mit der Umlaut-Umstellung in frontend/index.html und backend/templates/index.html.
        raise HTTPException(status_code=403, detail="E-Mail-Adresse noch nicht bestätigt. Bitte prüfen Sie Ihr Postfach oder fordern Sie einen neuen Bestätigungslink an.")

    # Update last_login
    conn = get_db()
    conn.execute("UPDATE users SET last_login = datetime('now') WHERE id = ?", (user["id"],))
    conn.commit()
    conn.close()
    token = create_token(user["id"], user["email"], user["is_admin"])
    response = JSONResponse({
        "ok": True,
        "email": user["email"],
        "display_name": user["display_name"],
        "is_admin": user["is_admin"],
    })
    response.set_cookie("token", token, httponly=True, samesite="strict", max_age=TOKEN_EXPIRE_HOURS * 3600)
    return response


@app.post("/api/register")
async def register(request: Request):
    # Registration can be disabled via environment variable
    if os.getenv("REGISTRATION_ENABLED", "true").lower() in ("false", "0", "no"):
        raise HTTPException(status_code=403, detail="Die Registrierung ist derzeit geschlossen. Bitte wende dich an kontakt@inklutec.de, um einen Testzugang zu erhalten.")
    data = await request.json()
    email = data.get("email", "").strip().lower()
    password = data.get("password", "")
    display_name = data.get("display_name", "").strip()

    if not email or "@" not in email:
        raise HTTPException(status_code=400, detail="Bitte eine gueltige E-Mail-Adresse eingeben")
    if len(password) < 8:
        raise HTTPException(status_code=400, detail="Passwort muss mindestens 8 Zeichen lang sein")
    if not display_name:
        raise HTTPException(status_code=400, detail="Bitte einen Namen eingeben")

    existing = get_user_by_email(email)
    if existing:
        raise HTTPException(status_code=409, detail="Diese E-Mail-Adresse ist bereits registriert")

    try:
        user_id = create_user(email, password, display_name, email_verified=0)
    except Exception as e:
        raise HTTPException(status_code=500, detail="Registrierung fehlgeschlagen")

    # Neue Nutzer starten bei den Neuigkeiten "auf dem aktuellen Stand". Bestandsnutzer
    # haben news_seen_until NULL und sehen daher beim naechsten Login alles Bisherige.
    try:
        _c = get_db()
        _c.execute("UPDATE users SET news_seen_until = ? WHERE id = ?", (_newest_news_key(), user_id))
        _c.commit()
        _c.close()
    except Exception:
        pass

    # Send verification email
    verify_token = create_email_verification_token(user_id, email)
    verify_url = f"{BASE_URL}/api/verify-email?token={verify_token}"
    email_body = f"""<!DOCTYPE html>
<html lang="de"><head><meta charset="utf-8"></head><body style="font-family:sans-serif;color:#1e293b;max-width:600px;margin:0 auto;">
<h1 style="color:#1b2a4a;">Willkommen bei InkluDocs</h1>
<p>Hallo {display_name},</p>
<p>vielen Dank fuer Ihre Registrierung bei InkluDocs – dem KI-gestuetzten Alt-Text-Generator fuer barrierefreie Dokumente und Bilder.</p>

<h2 style="color:#e87722;font-size:1.1rem;">E-Mail-Adresse bestaetigen</h2>
<p>Bitte klicken Sie auf den folgenden Link, um Ihr Konto zu aktivieren:</p>
<p><a href="{verify_url}" style="display:inline-block;background:#e87722;color:white;padding:0.75rem 1.5rem;border-radius:6px;text-decoration:none;font-weight:600;">E-Mail-Adresse bestaetigen</a></p>
<p style="color:#64748b;font-size:0.9rem;">Oder kopieren Sie diesen Link: {verify_url}</p>
<p style="color:#64748b;font-size:0.9rem;">Der Link ist 24 Stunden gueltig.</p>

<h2 style="color:#e87722;font-size:1.1rem;">So funktioniert InkluDocs</h2>
<p>Nachdem Sie Ihre E-Mail-Adresse bestaetigt haben:</p>
<ol style="line-height:1.8;">
<li>Melden Sie sich auf <a href="{BASE_URL}">{BASE_URL}</a> an</li>
<li>Laden Sie ein PDF, Bilder hoch oder geben Sie eine Website-URL ein</li>
<li>Klicken Sie auf &bdquo;Alt-Texte generieren&ldquo;</li>
<li>Bearbeiten Sie die Alt-Texte bei Bedarf und exportieren Sie sie</li>
</ol>

<p>Bei Fragen wenden Sie sich gerne an <a href="mailto:support@inklutec.de">support@inklutec.de</a>.</p>
<p style="color:#64748b;font-size:0.85rem;margin-top:2rem;">InkluDocs ist ein Produkt von INKLUTEC – support@inklutec.de</p>
</body></html>"""
    send_email(email, "InkluDocs: E-Mail-Adresse bestaetigen", email_body, bcc_admin=False)

    # Notify admin about new registration
    admin_body = f"""<!DOCTYPE html>
<html lang="de"><head><meta charset="utf-8"></head><body style="font-family:sans-serif;color:#1e293b;max-width:600px;">
<h2>Neue Registrierung bei InkluDocs</h2>
<p><strong>Name:</strong> {display_name}</p>
<p><strong>E-Mail:</strong> {email}</p>
<p><strong>Zeitpunkt:</strong> {datetime.utcnow().strftime("%d.%m.%Y %H:%M")} UTC</p>
<p style="color:#64748b;font-size:0.9rem;">Die E-Mail-Adresse wurde noch nicht bestätigt.</p>
</body></html>"""
    send_email(NOTIFICATION_EMAIL, "InkluDocs: Neue Registrierung", admin_body, bcc_admin=False)

    return JSONResponse({"ok": True, "message": "Bestätigungslink wurde gesendet. Bitte prüfen Sie Ihr Postfach."})


@app.post("/api/logout")
async def logout():
    response = JSONResponse({"ok": True})
    response.delete_cookie("token")
    return response


@app.get("/api/verify-email")
async def verify_email_registration(token: str = ""):
    """Verify email address from registration link."""
    if not token:
        return HTMLResponse("""<!DOCTYPE html><html lang="de"><head><meta charset="utf-8"><title>InkluDocs</title>
        <link rel="stylesheet" href="/static/style.css"></head><body>
        <div class="auth-wrapper"><div class="auth-container" role="main">
        <h1><span class="brand">Inklu</span>Docs</h1>
        <p style="color:#dc2626;margin:2rem 0;">Ungueltiger Bestaetigungslink.</p>
        <p><a href="/">Zur Anmeldung</a></p>
        </div></div></body></html>""", status_code=400)

    result = mark_email_verified(token)
    if not result:
        return HTMLResponse("""<!DOCTYPE html><html lang="de"><head><meta charset="utf-8"><title>InkluDocs</title>
        <link rel="stylesheet" href="/static/style.css"></head><body>
        <div class="auth-wrapper"><div class="auth-container" role="main">
        <h1><span class="brand">Inklu</span>Docs</h1>
        <p style="color:#dc2626;margin:2rem 0;">Dieser Bestaetigungslink ist ungueltig oder abgelaufen.</p>
        <p><a href="/">Zur Anmeldung</a></p>
        </div></div></body></html>""", status_code=400)

    # Notify admin that email was confirmed
    admin_body = f"""<!DOCTYPE html>
<html lang="de"><head><meta charset="utf-8"></head><body style="font-family:sans-serif;color:#1e293b;max-width:600px;">
<h2 style="color:#16a34a;">E-Mail bestaetigt</h2>
<p><strong>E-Mail:</strong> {result['email']}</p>
<p>Der Benutzer kann sich jetzt anmelden.</p>
</body></html>"""
    send_email(NOTIFICATION_EMAIL, "InkluDocs: E-Mail bestaetigt", admin_body, bcc_admin=False)

    return HTMLResponse(f"""<!DOCTYPE html><html lang="de"><head><meta charset="utf-8"><title>InkluDocs – E-Mail bestaetigt</title>
    <link rel="stylesheet" href="/static/style.css"></head><body>
    <div class="auth-wrapper"><div class="auth-container" role="main" aria-label="E-Mail bestaetigt">
    <h1><span class="brand">Inklu</span>Docs</h1>
    <p style="color:#16a34a;font-size:1.2rem;margin:2rem 0;font-weight:600;">&#10003; E-Mail-Adresse erfolgreich bestaetigt!</p>
    <p>Ihr Konto ist jetzt aktiv. Sie koennen sich jetzt anmelden.</p>
    <p style="margin-top:1.5rem;"><a href="/" style="display:inline-block;background:#e87722;color:white;padding:0.75rem 1.5rem;border-radius:6px;text-decoration:none;font-weight:600;">Jetzt anmelden</a></p>
    </div></div></body></html>""")


@app.post("/api/resend-verification")
async def resend_verification(request: Request):
    """Resend verification email for unverified accounts."""
    data = await request.json()
    email = data.get("email", "").strip().lower()
    if not email:
        raise HTTPException(status_code=400, detail="E-Mail-Adresse erforderlich")

    token = resend_verification_token(email)
    if not token:
        # Don't reveal whether the email exists or is already verified
        return JSONResponse({"ok": True, "message": "Falls ein unverifiziertes Konto existiert, wurde eine neue Bestaetigungsmail gesendet."})

    user = get_user_by_email(email)
    display_name = user["display_name"] if user else ""
    verify_url = f"{BASE_URL}/api/verify-email?token={token}"
    email_body = f"""<!DOCTYPE html>
<html lang="de"><head><meta charset="utf-8"></head><body style="font-family:sans-serif;color:#1e293b;max-width:600px;">
<h2 style="color:#e87722;">InkluDocs – E-Mail bestaetigen</h2>
<p>Hallo {display_name},</p>
<p>hier ist Ihr neuer Bestaetigungslink:</p>
<p><a href="{verify_url}" style="display:inline-block;background:#e87722;color:white;padding:0.75rem 1.5rem;border-radius:6px;text-decoration:none;font-weight:600;">E-Mail-Adresse bestaetigen</a></p>
<p style="color:#64748b;font-size:0.9rem;">Oder kopieren Sie diesen Link: {verify_url}</p>
<p style="color:#64748b;font-size:0.9rem;">Der Link ist 24 Stunden gueltig.</p>
<p style="color:#64748b;font-size:0.85rem;margin-top:2rem;">InkluDocs – support@inklutec.de</p>
</body></html>"""
    send_email(email, "InkluDocs: E-Mail-Adresse bestaetigen", email_body, bcc_admin=False)

    return JSONResponse({"ok": True, "message": "Falls ein unverifiziertes Konto existiert, wurde eine neue Bestaetigungsmail gesendet."})


@app.get("/api/me")
async def me(user: dict = Depends(get_current_user)):
    db_user = get_user_by_id(user["id"])
    if not db_user:
        raise HTTPException(status_code=401, detail="User nicht gefunden")
    daily_used = get_daily_image_count(db_user["id"])
    return {
        "ok": True,
        "user": {
            "id": db_user["id"],
            "email": db_user["email"],
            "display_name": db_user["display_name"],
            "is_admin": db_user["is_admin"],
            "admin_level": db_user.get("admin_level", "full"),
        },
        "daily_limit": {
            "used": daily_used,
            "limit": DAILY_IMAGE_LIMIT,
            "remaining": max(0, DAILY_IMAGE_LIMIT - daily_used),
        },
    }


@app.post("/api/change-password")
async def change_password(request: Request, user: dict = Depends(get_current_user)):
    data = await request.json()
    old_password = data.get("old_password", "")
    new_password = data.get("new_password", "")

    if len(new_password) < 8:
        raise HTTPException(status_code=400, detail="Neues Passwort muss mindestens 8 Zeichen lang sein")

    db_user = get_user_by_id(user["id"])
    verified = verify_user(db_user["email"], old_password)
    if not verified:
        raise HTTPException(status_code=401, detail="Aktuelles Passwort ist falsch")

    admin_reset_password(user["id"], new_password)
    return {"ok": True, "message": "Passwort wurde geaendert"}


@app.post("/api/change-email")
async def change_email(request: Request, user: dict = Depends(get_current_user)):
    """Request email change – sends confirmation link to new address."""
    data = await request.json()
    new_email = data.get("new_email", "").strip().lower()

    if not new_email or "@" not in new_email:
        raise HTTPException(status_code=400, detail="Bitte eine gueltige E-Mail-Adresse eingeben")

    if new_email == user["email"]:
        raise HTTPException(status_code=400, detail="Das ist bereits deine aktuelle E-Mail-Adresse")

    existing = get_user_by_email(new_email)
    if existing:
        raise HTTPException(status_code=409, detail="Diese E-Mail-Adresse ist bereits vergeben")

    db_user = get_user_by_id(user["id"])
    token = create_email_change_token(user["id"], new_email)
    confirm_url = f"{BASE_URL}/api/confirm-email?token={token}"

    email_body = f"""<!DOCTYPE html>
<html lang="de"><head><meta charset="utf-8"></head><body style="font-family:sans-serif;color:#1e293b;max-width:600px;">
<h1 style="color:#1b2a4a;">E-Mail-Adresse bestaetigen</h1>
<p>Hallo {db_user['display_name']},</p>
<p>du hast angefordert, deine InkluDocs E-Mail-Adresse auf <strong>{new_email}</strong> zu aendern.</p>
<p><a href="{confirm_url}" style="display:inline-block;background:#e87722;color:white;padding:0.75rem 1.5rem;border-radius:6px;text-decoration:none;font-weight:600;">E-Mail-Adresse bestaetigen</a></p>
<p style="color:#64748b;font-size:0.9rem;">Oder kopiere diesen Link: {confirm_url}</p>
<p style="color:#64748b;font-size:0.9rem;">Der Link ist 1 Stunde gueltig. Falls du diese Aenderung nicht angefordert hast, ignoriere diese E-Mail.</p>
<p style="color:#64748b;font-size:0.85rem;margin-top:2rem;">InkluDocs – kontakt@inklutec.de</p>
</body></html>"""

    sent = send_email(new_email, "InkluDocs: E-Mail-Adresse bestaetigen", email_body, bcc_admin=False)
    if not sent:
        raise HTTPException(status_code=500, detail="Bestaetigungsmail konnte nicht gesendet werden. Bitte spaeter erneut versuchen.")

    return {"ok": True, "message": f"Bestaetigungslink wurde an {new_email} gesendet. Bitte pruefe dein Postfach."}


@app.get("/api/confirm-email")
async def confirm_email(token: str = ""):
    """Confirm email change via link from confirmation email."""
    if not token:
        raise HTTPException(status_code=400, detail="Token fehlt")

    result = confirm_email_change(token)
    if not result:
        return HTMLResponse("""<!DOCTYPE html><html lang="de"><head><meta charset="utf-8"><title>InkluDocs</title>
<link rel="stylesheet" href="/static/style.css"></head><body>
<div class="auth-wrapper"><div class="auth-container" role="main" aria-label="E-Mail-Bestaetigung fehlgeschlagen">
<h1><span class="brand">Inklu</span>Docs</h1>
<p style="margin:2rem 0;color:var(--error,#dc2626);font-weight:600;">Der Bestaetigungslink ist ungueltig oder abgelaufen.</p>
<p>Bitte fordere in den <a href="/app">Einstellungen</a> einen neuen Link an.</p>
</div></div></body></html>""", status_code=400)

    # Success – show confirmation page and auto-redirect
    db_user = get_user_by_id(result["user_id"])
    return HTMLResponse(f"""<!DOCTYPE html><html lang="de"><head><meta charset="utf-8"><title>InkluDocs</title>
<link rel="stylesheet" href="/static/style.css"><meta http-equiv="refresh" content="3;url=/app"></head><body>
<div class="auth-wrapper"><div class="auth-container" role="main" aria-label="E-Mail-Adresse bestaetigt">
<h1><span class="brand">Inklu</span>Docs</h1>
<p style="margin:2rem 0;color:var(--success,#16a34a);font-weight:600;">Deine E-Mail-Adresse wurde erfolgreich auf {result['new_email']} geaendert.</p>
<p>Du wirst in 3 Sekunden weitergeleitet. <a href="/app">Jetzt zur App</a></p>
</div></div></body></html>""")


@app.post("/api/change-displayname")
async def change_displayname(request: Request, user: dict = Depends(get_current_user)):
    data = await request.json()
    new_name = data.get("display_name", "").strip()

    if not new_name:
        raise HTTPException(status_code=400, detail="Bitte einen Namen eingeben")
    if len(new_name) > 100:
        raise HTTPException(status_code=400, detail="Name darf maximal 100 Zeichen lang sein")

    conn = get_db()
    conn.execute("UPDATE users SET display_name = ? WHERE id = ?", (new_name, user["id"]))
    conn.commit()
    conn.close()
    return {"ok": True, "message": "Name wurde geaendert", "display_name": new_name}


# ─── Password Reset ──────────────────────────────────────────

@app.post("/api/forgot-password")
async def forgot_password(request: Request):
    data = await request.json()
    email = data.get("email", "").strip().lower()
    user = get_user_by_email(email)

    # Always return success (don't reveal if email exists)
    if not user:
        return {"ok": True, "message": "Falls ein Konto existiert, wird ein Reset-Link angezeigt."}

    token = create_password_reset_token(user["id"])
    reset_url = f"{BASE_URL}/reset?token={token}"

    # Send reset link via email (never expose in API response)
    email_body = f"""<!DOCTYPE html>
<html lang="de"><head><meta charset="utf-8"></head><body style="font-family:sans-serif;color:#1e293b;max-width:600px;">
<h1 style="color:#1b2a4a;">Passwort zurücksetzen</h1>
<p>Hallo {user['display_name']},</p>
<p>du hast eine Passwort-Zurücksetzung für dein InkluDocs-Konto angefordert.</p>
<p><a href="{reset_url}" style="display:inline-block;background:#e87722;color:white;padding:0.75rem 1.5rem;border-radius:6px;text-decoration:none;font-weight:600;">Passwort jetzt zurücksetzen</a></p>
<p style="color:#64748b;font-size:0.9rem;">Oder kopiere diesen Link: {reset_url}</p>
<p style="color:#64748b;font-size:0.9rem;">Der Link ist 1 Stunde gueltig. Falls du diese Anfrage nicht gestellt hast, kannst du diese E-Mail ignorieren.</p>
<p style="color:#64748b;font-size:0.85rem;margin-top:2rem;">InkluDocs – kontakt@inklutec.de</p>
</body></html>"""
    send_email(email, "InkluDocs: Passwort zurücksetzen", email_body, bcc_admin=False)

    return {
        "ok": True,
        "message": "Falls ein Konto mit dieser E-Mail existiert, wurde ein Reset-Link per E-Mail gesendet.",
    }


@app.post("/api/reset-password")
async def do_reset_password(request: Request):
    data = await request.json()
    token = data.get("token", "")
    new_password = data.get("new_password", "")

    if len(new_password) < 8:
        raise HTTPException(status_code=400, detail="Passwort muss mindestens 8 Zeichen lang sein")

    if not reset_password(token, new_password):
        raise HTTPException(status_code=400, detail="Reset-Link ist ungueltig oder abgelaufen")

    return {"ok": True, "message": "Passwort wurde zurueckgesetzt. Sie koennen sich jetzt anmelden."}


# ─── Admin Routes ────────────────────────────────────────────

@app.get("/api/admin/users")
async def admin_list_users(user: dict = Depends(require_admin)):
    users = list_all_users()
    # Count projects per user
    conn = get_db()
    for u in users:
        row = conn.execute("SELECT COUNT(*) as cnt FROM projects WHERE user_id = ?", (u["id"],)).fetchone()
        u["project_count"] = row["cnt"]
    conn.close()
    return {"users": users}


@app.post("/api/admin/users/{user_id}/toggle-active")
async def admin_toggle_active(user_id: int, user: dict = Depends(require_full_admin)):
    target = get_user_by_id(user_id)
    if not target:
        raise HTTPException(status_code=404, detail="User nicht gefunden")
    if target["id"] == user["id"]:
        raise HTTPException(status_code=400, detail="Sie koennen sich nicht selbst deaktivieren")
    new_status = 0 if target["is_active"] else 1
    update_user_active(user_id, new_status)
    return {"ok": True, "is_active": new_status}


@app.post("/api/admin/users/{user_id}/reset-password")
async def admin_reset_user_password(user_id: int, request: Request, user: dict = Depends(require_full_admin)):
    data = await request.json()
    new_password = data.get("new_password", "")
    if len(new_password) < 8:
        raise HTTPException(status_code=400, detail="Passwort muss mindestens 8 Zeichen lang sein")
    target = get_user_by_id(user_id)
    if not target:
        raise HTTPException(status_code=404, detail="User nicht gefunden")
    admin_reset_password(user_id, new_password)
    return {"ok": True, "message": f"Passwort fuer {target['email']} wurde zurueckgesetzt"}


@app.post("/api/admin/users/create")
async def admin_create_user(request: Request, user: dict = Depends(require_full_admin)):
    """Admin: Create a new user account."""
    data = await request.json()
    email = data.get("email", "").strip().lower()
    password = data.get("password", "")
    display_name = data.get("display_name", "").strip()

    if not email or "@" not in email:
        raise HTTPException(status_code=400, detail="Bitte eine gueltige E-Mail-Adresse eingeben")
    if len(password) < 8:
        raise HTTPException(status_code=400, detail="Passwort muss mindestens 8 Zeichen lang sein")
    if not display_name:
        raise HTTPException(status_code=400, detail="Bitte einen Namen eingeben")

    existing = get_user_by_email(email)
    if existing:
        raise HTTPException(status_code=409, detail="Diese E-Mail-Adresse ist bereits registriert")

    try:
        user_id = create_user(email, password, display_name)
    except Exception:
        raise HTTPException(status_code=500, detail="Benutzer konnte nicht erstellt werden")

    # Send welcome email with credentials
    send_welcome = data.get("send_email", True)
    email_sent = False
    if send_welcome:
        email_body = f"""<!DOCTYPE html>
<html lang="de"><head><meta charset="utf-8"></head><body style="font-family:sans-serif;color:#1e293b;max-width:600px;margin:0 auto;">
<h1 style="color:#1b2a4a;">Willkommen bei InkluDocs</h1>
<p>Hallo {display_name},</p>
<p>dein Zugang zu InkluDocs wurde erstellt. InkluDocs ist ein KI-gestützter Alt-Text-Generator für barrierefreie Dokumente und Bilder.</p>
<h2 style="color:#e87722;font-size:1.1rem;">Deine Zugangsdaten</h2>
<p><strong>Login-Seite:</strong> <a href="{BASE_URL}">{BASE_URL}</a></p>
<p><strong>E-Mail:</strong> {email}</p>
<p><strong>Passwort:</strong> {password}</p>
<p style="background:#fff7ed;padding:1rem;border-left:3px solid #e87722;border-radius:0 4px 4px 0;">
Bitte ändere dein Passwort nach dem ersten Login unter <strong>Einstellungen</strong>.</p>
<h2 style="color:#e87722;font-size:1.1rem;">So funktioniert es</h2>
<ol>
<li>Melde dich auf <a href="{BASE_URL}">{BASE_URL}</a> an</li>
<li>Lade ein PDF, Bilder hoch oder gib eine Website-URL ein</li>
<li>Klicke auf "Alt-Texte generieren"</li>
<li>Bearbeite die Alt-Texte bei Bedarf und exportiere sie</li>
</ol>
<p>Bei Fragen wende dich an <a href="mailto:kontakt@inklutec.de">kontakt@inklutec.de</a>.</p>
<p style="color:#64748b;font-size:0.85rem;margin-top:2rem;">InkluDocs ist ein Produkt von INKLUTEC – kontakt@inklutec.de</p>
</body></html>"""
        email_sent = send_email(email, f"Dein InkluDocs-Zugang", email_body)

    msg = f"Benutzer {display_name} ({email}) wurde erstellt"
    if email_sent:
        msg += " – Zugangsdaten per E-Mail gesendet"
    else:
        msg += " – Zugangsdaten bitte manuell weitergeben"

    return {"ok": True, "message": msg, "user_id": user_id, "email_sent": email_sent}


@app.delete("/api/admin/users/{user_id}")
async def admin_delete_user(user_id: int, user: dict = Depends(require_full_admin)):
    target = get_user_by_id(user_id)
    if not target:
        raise HTTPException(status_code=404, detail="User nicht gefunden")
    if target["id"] == user["id"]:
        raise HTTPException(status_code=400, detail="Sie koennen sich nicht selbst loeschen")
    # Delete user files from disk
    user_upload_dir = os.path.join(UPLOAD_DIR, str(user_id))
    user_results_dir = os.path.join(RESULTS_DIR, str(user_id))
    if os.path.exists(user_upload_dir):
        shutil.rmtree(user_upload_dir)
    if os.path.exists(user_results_dir):
        shutil.rmtree(user_results_dir)
    # Delete from DB (DSGVO-konform: alle Daten werden geloescht)
    delete_user_data(user_id)
    return {"ok": True, "message": f"User {target['email']} und alle Daten wurden geloescht"}


# ─── Administrator-Verwaltung (14.06.2026) ──────────────────
# Voll-Admins koennen bestehende Konten zu Admins machen und die
# Rechte-Stufe setzen ('full' = alle Rechte, 'view' = nur Einsicht).
# Sicherungen: niemand kann sich selbst die Admin-Rechte entziehen,
# und der letzte Voll-Admin kann weder entfernt noch herabgestuft
# werden (Schutz gegen Aussperren).

@app.get("/api/admin/admins")
async def admin_list_admins(user: dict = Depends(require_admin)):
    """Liste aller Administratoren inkl. Rechte-Stufe (auch fuer Nur-Einsicht lesbar)."""
    return {"admins": list_admins()}


@app.post("/api/admin/admins")
async def admin_add_admin(request: Request, user: dict = Depends(require_full_admin)):
    """Ein bestehendes Konto zum Admin machen (per E-Mail) und Stufe setzen."""
    data = await request.json()
    email = data.get("email", "").strip().lower()
    level = data.get("level", "full")
    if level not in ("full", "view"):
        raise HTTPException(status_code=400, detail="Ungueltige Rechte-Stufe")
    if not email or "@" not in email:
        raise HTTPException(status_code=400, detail="Bitte eine gueltige E-Mail-Adresse eingeben")
    target = get_user_by_email(email)
    if not target:
        raise HTTPException(status_code=404, detail="Kein Konto mit dieser E-Mail. Bitte die Person zuerst unter 'Neuen Benutzer anlegen' anlegen und danach zum Admin machen.")
    if target["is_admin"]:
        raise HTTPException(status_code=409, detail=f"{target['display_name']} ist bereits Administrator")
    set_user_admin(target["id"], 1, level)
    stufe = "Voll-Admin" if level == "full" else "Nur-Einsicht"
    return {"ok": True, "message": f"{target['display_name']} ist jetzt Administrator ({stufe})"}


@app.put("/api/admin/admins/{user_id}")
async def admin_set_level(user_id: int, request: Request, user: dict = Depends(require_full_admin)):
    """Rechte-Stufe eines Admins aendern (full <-> view)."""
    data = await request.json()
    level = data.get("level", "")
    if level not in ("full", "view"):
        raise HTTPException(status_code=400, detail="Ungueltige Rechte-Stufe")
    target = get_user_by_id(user_id)
    if not target or not target["is_admin"]:
        raise HTTPException(status_code=404, detail="Administrator nicht gefunden")
    if level == "view" and target["admin_level"] == "full" and count_full_admins() <= 1:
        raise HTTPException(status_code=400, detail="Der letzte Voll-Admin kann nicht auf 'Nur-Einsicht' gesetzt werden")
    set_user_admin(user_id, 1, level)
    stufe = "Voll-Admin" if level == "full" else "Nur-Einsicht"
    return {"ok": True, "message": f"{target['display_name']} ist jetzt {stufe}", "level": level}


@app.delete("/api/admin/admins/{user_id}")
async def admin_revoke_admin(user_id: int, user: dict = Depends(require_full_admin)):
    """Admin-Rechte entziehen (das Konto bleibt als normaler Nutzer bestehen)."""
    target = get_user_by_id(user_id)
    if not target or not target["is_admin"]:
        raise HTTPException(status_code=404, detail="Administrator nicht gefunden")
    if target["id"] == user["id"]:
        raise HTTPException(status_code=400, detail="Sie koennen sich die Admin-Rechte nicht selbst entziehen")
    if target["admin_level"] == "full" and count_full_admins() <= 1:
        raise HTTPException(status_code=400, detail="Der letzte Voll-Admin kann nicht entfernt werden")
    set_user_admin(user_id, 0)
    return {"ok": True, "message": f"Admin-Rechte von {target['display_name']} entzogen"}


# ─── API Key Management ─────────────────────────────────────

@app.get("/api/api-keys")
async def api_list_keys(user: dict = Depends(get_current_user)):
    keys = list_api_keys(user["id"])
    return {"api_keys": keys}


@app.post("/api/api-keys")
async def api_create_key(request: Request, user: dict = Depends(get_current_user)):
    data = await request.json()
    name = data.get("name", "").strip()
    if not name:
        raise HTTPException(status_code=400, detail="Bitte einen Namen fuer den API-Key eingeben")
    key_id, raw_key = create_api_key(user["id"], name)
    return {
        "ok": True,
        "key_id": key_id,
        "api_key": raw_key,
        "hinweis": "Dieser API-Key wird nur einmal angezeigt. Bitte sicher speichern.",
    }


@app.put("/api/api-keys/{key_id}")
async def api_rename_key(key_id: int, request: Request, user: dict = Depends(get_current_user)):
    data = await request.json()
    new_name = data.get("name", "").strip()
    if not new_name:
        raise HTTPException(status_code=400, detail="Bitte einen Namen eingeben")
    if not rename_api_key(user["id"], key_id, new_name):
        raise HTTPException(status_code=404, detail="API-Key nicht gefunden")
    return {"ok": True}


@app.delete("/api/api-keys/{key_id}")
async def api_delete_key(key_id: int, user: dict = Depends(get_current_user)):
    if not delete_api_key(user["id"], key_id):
        raise HTTPException(status_code=404, detail="API-Key nicht gefunden")
    return {"ok": True}


# ─── Project Routes ──────────────────────────────────────────

@app.get("/api/projects")
async def list_projects(user: dict = Depends(get_current_user)):
    conn = get_db()
    rows = conn.execute(
        "SELECT * FROM projects WHERE user_id = ? ORDER BY created_at DESC", (user["id"],)
    ).fetchall()
    conn.close()
    return {"projects": [dict(r) for r in rows]}


@app.post("/api/projects")
async def create_project(request: Request, user: dict = Depends(get_current_user)):
    """Legt ein leeres Projekt an (Name + Werkzeug). Die Datei(en) werden
    anschliessend im jeweiligen Werkzeug hochgeladen (siehe Upload-Endpunkte)."""
    data = await request.json()
    name = (data.get("name") or "").strip()
    tool = (data.get("tool") or "").strip()
    if not name:
        raise HTTPException(status_code=400, detail="Bitte einen Projektnamen eingeben")
    if len(name) > 120:
        raise HTTPException(status_code=400, detail="Projektname darf maximal 120 Zeichen lang sein")
    if not is_valid_tool_key(tool):
        raise HTTPException(status_code=400, detail="Unbekanntes oder nicht verfuegbares Werkzeug")
    project_type = {"pdf": "pdf", "web": "url", "grafik": "images"}.get(tool)
    if not project_type:
        raise HTTPException(status_code=400, detail="Fuer dieses Werkzeug ist das Anlegen noch nicht verfuegbar")
    conn = get_db()
    cursor = conn.execute(
        "INSERT INTO projects (user_id, name, filename, original_path, status, project_type, tool) VALUES (?, ?, ?, '', 'neu', ?, ?)",
        (user["id"], name, name, project_type, tool)
    )
    project_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return {"ok": True, "project_id": project_id, "tool": tool, "project_type": project_type}


@app.patch("/api/projects/{project_id}")
async def rename_project(project_id: int, request: Request, user: dict = Depends(get_current_user)):
    """Benennt ein Projekt um (freier Anzeigename)."""
    data = await request.json()
    name = (data.get("name") or "").strip()
    if not name:
        raise HTTPException(status_code=400, detail="Bitte einen Projektnamen eingeben")
    if len(name) > 120:
        raise HTTPException(status_code=400, detail="Projektname darf maximal 120 Zeichen lang sein")
    conn = get_db()
    cur = conn.execute("UPDATE projects SET name = ? WHERE id = ? AND user_id = ?", (name, project_id, user["id"]))
    conn.commit()
    affected = cur.rowcount
    conn.close()
    if not affected:
        raise HTTPException(status_code=404, detail="Projekt nicht gefunden")
    return {"ok": True, "name": name}


@app.post("/api/projects/{project_id}/context-setting")
async def set_context_setting(project_id: int, request: Request, user: dict = Depends(get_current_user)):
    """Setzt den Projekt-Schalter 'KI-Kontext verwenden' (19.06.2026).

    use_context=1 -> kommende Generierungen geben der KI den Dokument-Kontext
    (Seiten-/Kapiteltext); =0 -> die KI bekommt keinen Kontext. Wirkt nur auf
    NEU generierte Bilder; bestehende bleiben, bis man sie neu generiert.
    """
    data = await request.json()
    use_ctx = 1 if data.get("use_context") else 0
    conn = get_db()
    cur = conn.execute(
        "UPDATE projects SET use_context = ? WHERE id = ? AND user_id = ?",
        (use_ctx, project_id, user["id"])
    )
    conn.commit()
    affected = cur.rowcount
    conn.close()
    if not affected:
        raise HTTPException(status_code=404, detail="Projekt nicht gefunden")
    return {"ok": True, "use_context": bool(use_ctx)}


@app.get("/api/tools")
async def list_tools(user: dict = Depends(get_current_user)):
    """Liefert die im Dashboard angebotenen Werkzeuge (Quelle: tools.py)."""
    return {
        "tools": [
            {
                "key": t.key,
                "name": t.name,
                "description": t.description,
                "route": t.route,
                "status": t.status.value,
                "status_label": t.status_label,
                "is_available": t.is_available,
            }
            for t in TOOLS
        ]
    }


@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...), project_id: int = Form(None), user: dict = Depends(get_current_user)):
    """Upload a PDF or image file(s). Accepts PDF, JPG, JPEG, PNG, GIF, SVG, WEBP."""
    filename = file.filename or "unknown"
    ext = os.path.splitext(filename)[1].lower()

    is_pdf = ext == ".pdf"
    is_image = ext in IMAGE_EXTENSIONS

    if not is_pdf and not is_image:
        raise HTTPException(
            status_code=400,
            detail="Nur PDF- und Bilddateien erlaubt (PDF, JPG, PNG, GIF, SVG, WebP, HEIC, BMP, TIFF)"
        )

    # Read and check file size
    content = await file.read()
    if len(content) > MAX_UPLOAD_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"Datei zu gross. Maximum: {MAX_UPLOAD_SIZE // (1024*1024)} MB"
        )

    # Create user directory
    user_dir = os.path.join(UPLOAD_DIR, str(user["id"]))
    os.makedirs(user_dir, exist_ok=True)

    # Save file with sanitized name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = f"{timestamp}_{os.path.basename(filename)}"
    file_path = os.path.join(user_dir, safe_name)
    with open(file_path, "wb") as f:
        f.write(content)

    if is_pdf:
        return await _handle_pdf_upload(file_path, filename, user, project_id)
    else:
        return await _handle_image_upload(file_path, filename, user, content, ext, project_id)


async def _handle_pdf_upload(file_path: str, filename: str, user: dict, project_id: int = None) -> dict:
    """Legt ein neues PDF-Projekt an ODER haengt eine weitere PDF an ein
    bestehendes PDF-Projekt an.

    Multi-Datei Phase 1 (08.06.2026): Ein PDF-Projekt kann mehrere PDFs
    enthalten. Jede hochgeladene PDF wird zu einem Dokument
    (Tabelle documents). Bilder bekommen die document_id und ihre
    page_number bleibt die PDF-interne Seitennummer — Kollisionen zwischen
    Dokumenten loest die document_id auf. Sammel-Generierung verarbeitet
    weiterhin nur status='pending' (siehe _process_project), ueberschreibt
    also bereits fertige Alt-Texte der vorherigen Dokumente NICHT.
    """
    conn = get_db()
    is_append = False  # zeigt an, ob wir an ein bestehendes PDF-Projekt anhaengen
    if project_id is not None:
        proj = conn.execute(
            "SELECT id, tool, total_images FROM projects WHERE id = ? AND user_id = ?",
            (project_id, user["id"])
        ).fetchone()
        if not proj:
            conn.close()
            raise HTTPException(status_code=404, detail="Projekt nicht gefunden")
        if proj["tool"] != "pdf":
            conn.close()
            raise HTTPException(status_code=400, detail="Dieses Projekt ist kein PDF-Projekt")
        # Multi-Datei: Anhaengen ist ab jetzt erlaubt — kein 409 mehr bei total_images > 0.
        is_append = bool(proj["total_images"] and proj["total_images"] > 0)
        # status auf extracting setzen; filename/original_path nur beim ERSTEN
        # PDF setzen, damit die "Hauptdatei" des Projekts (z.B. fuer den Projektkopf)
        # erhalten bleibt. Backfill setzt das ohnehin auf das erste Dokument.
        if is_append:
            conn.execute("UPDATE projects SET status = 'extracting' WHERE id = ?", (project_id,))
        else:
            conn.execute(
                "UPDATE projects SET filename = ?, original_path = ?, status = 'extracting' WHERE id = ?",
                (filename, file_path, project_id)
            )
        conn.commit()
    else:
        cursor = conn.execute(
            "INSERT INTO projects (user_id, filename, original_path, status, project_type, tool) VALUES (?, ?, ?, 'extracting', 'pdf', 'pdf')",
            (user["id"], filename, file_path)
        )
        project_id = cursor.lastrowid
        conn.commit()

    # Naechsten doc_index ermitteln (1-basiert, fortlaufend pro Projekt).
    doc_index = (conn.execute(
        "SELECT COALESCE(MAX(doc_index), 0) FROM documents WHERE project_id = ?",
        (project_id,)
    ).fetchone()[0] or 0) + 1

    # Dokument-Eintrag anlegen (extraction_method wird unten nachgetragen).
    doc_cursor = conn.execute(
        """INSERT INTO documents
           (project_id, doc_index, original_filename, original_path, extraction_method, total_images)
           VALUES (?, ?, ?, ?, ?, 0)""",
        (project_id, doc_index, filename, file_path, "fitz")
    )
    document_id = doc_cursor.lastrowid
    conn.commit()

    # Extract images — eigener Unterordner pro Dokument, damit sich die
    # PNG-Dateinamen (p1_img1.png, ...) zwischen Dokumenten nicht ueberschreiben.
    img_dir = os.path.join(RESULTS_DIR, str(user["id"]), str(project_id), f"doc{doc_index}")
    os.makedirs(img_dir, exist_ok=True)

    try:
        images = extract_images_from_pdf(file_path, img_dir, project_id)
    except Exception as e:
        # Nur das Dokument als fehlerhaft markieren; das Projekt darf weiterlaufen,
        # wenn frueher schon Dokumente erfolgreich extrahiert wurden.
        conn.execute("DELETE FROM documents WHERE id = ?", (document_id,))
        if not is_append:
            conn.execute("UPDATE projects SET status = 'error' WHERE id = ?", (project_id,))
        conn.commit()
        conn.close()
        raise HTTPException(status_code=500, detail=f"PDF-Verarbeitung fehlgeschlagen: {str(e)}")

    # Naechsten image_index fortlaufen lassen, damit alte Projekte weiter
    # konsistent durchnummeriert bleiben. Das pro-Dokument-Nummer-Feld leistet
    # die Anzeige im Frontend (dort einfach Position innerhalb des Dokuments).
    next_idx = conn.execute(
        "SELECT COALESCE(MAX(image_index), 0) FROM images WHERE project_id = ?", (project_id,)
    ).fetchone()[0] or 0

    for img in images:
        bbox = img.get("bbox")
        bbox_x0 = bbox[0] if bbox else None
        bbox_y0 = bbox[1] if bbox else None
        bbox_x1 = bbox[2] if bbox else None
        bbox_y1 = bbox[3] if bbox else None
        is_vector = 1 if img.get("is_vector") else 0
        next_idx += 1

        conn.execute(
            """INSERT INTO images (project_id, document_id, page_number, image_index, image_path, context_text,
               width, height, xref, bbox_x0, bbox_y0, bbox_x1, bbox_y1, is_vector,
               original_alt, page_view_path, page_text)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (project_id, document_id, img["page_number"], next_idx, img["image_path"],
             img["context_text"], img["width"], img["height"], img["xref"],
             bbox_x0, bbox_y0, bbox_x1, bbox_y1, is_vector,
             img.get("original_alt", ""),
             img.get("page_view_path", ""), img.get("page_text", ""))
        )

    # PDFIX-INTEGRATION (24.04.2026): Extraktionsweg merken (fitz|pdfix)
    extraction_method = "pdfix" if images and any(i.get("source") == "pdfix" for i in images) else "fitz"
    conn.execute(
        "UPDATE documents SET extraction_method = ?, total_images = ? WHERE id = ?",
        (extraction_method, len(images), document_id)
    )
    # Projekt-Summe = Summe ueber alle Dokumente. extraction_method des Projekts
    # bleibt der des ersten Dokuments (Anzeige im Kopf) — bei Mischfaellen sind
    # die Methoden je Dokument im UI sichtbar.
    total_imgs = conn.execute(
        "SELECT COUNT(*) FROM images WHERE project_id = ?", (project_id,)
    ).fetchone()[0]
    if is_append:
        conn.execute(
            "UPDATE projects SET status = 'extracted', total_images = ? WHERE id = ?",
            (total_imgs, project_id)
        )
    else:
        conn.execute(
            "UPDATE projects SET status = 'extracted', total_images = ?, extraction_method = ? WHERE id = ?",
            (total_imgs, extraction_method, project_id)
        )
    conn.commit()
    conn.close()

    return {
        "ok": True,
        "project_id": project_id,
        "document_id": document_id,
        "doc_index": doc_index,
        "filename": filename,
        "total_images": total_imgs,
        "added_images": len(images),
        "project_type": "pdf",
        "extraction_method": extraction_method,
        "appended": is_append,
    }


async def _handle_image_upload(file_path: str, filename: str, user: dict, content: bytes, ext: str, project_id: int = None) -> dict:
    """Process a direct image upload."""
    from PIL import Image as PILImage
    try:
        from pillow_heif import register_heif_opener
        register_heif_opener()
    except ImportError:
        pass

    conn = get_db()
    if project_id is not None:
        proj = conn.execute(
            "SELECT id, tool FROM projects WHERE id = ? AND user_id = ?",
            (project_id, user["id"])
        ).fetchone()
        if not proj:
            conn.close()
            raise HTTPException(status_code=404, detail="Projekt nicht gefunden")
        if proj["tool"] != "grafik":
            conn.close()
            raise HTTPException(status_code=400, detail="Dieses Projekt ist kein Grafik-Projekt")
    else:
        cursor = conn.execute(
            "INSERT INTO projects (user_id, filename, original_path, status, project_type, tool) VALUES (?, ?, ?, 'extracted', 'images', 'grafik')",
            (user["id"], filename, file_path)
        )
        project_id = cursor.lastrowid
        conn.commit()
    # Fortlaufende Bild-Nummer (auch beim Anhaengen weiterer Bilder)
    next_idx = conn.execute("SELECT COALESCE(MAX(image_index), 0) FROM images WHERE project_id = ?", (project_id,)).fetchone()[0] + 1

    # Create project image directory
    img_dir = os.path.join(RESULTS_DIR, str(user["id"]), str(project_id))
    os.makedirs(img_dir, exist_ok=True)

    # Convert HEIC/HEIF/BMP/TIFF to JPEG for model compatibility
    needs_conversion = ext in {".heic", ".heif", ".bmp", ".tiff", ".tif"}
    if needs_conversion:
        img_path = os.path.join(img_dir, f"img_{next_idx}.jpg")
        try:
            with PILImage.open(file_path) as img:
                rgb_img = img.convert("RGB")
                rgb_img.save(img_path, format="JPEG", quality=90)
        except Exception as e:
            conn.execute("UPDATE projects SET status = 'error' WHERE id = ?", (project_id,))
            conn.commit()
            conn.close()
            raise HTTPException(status_code=400, detail=f"Bildformat konnte nicht konvertiert werden: {str(e)}")
    else:
        img_path = os.path.join(img_dir, f"img_{next_idx}{ext}")
        shutil.copy2(file_path, img_path)

    # Get image dimensions
    width, height = 0, 0
    try:
        with PILImage.open(img_path) as img:
            width, height = img.size
    except Exception:
        pass

    conn.execute(
        """INSERT INTO images (project_id, page_number, image_index, image_path, context_text,
           width, height, xref)
           VALUES (?, 1, ?, ?, '', ?, ?, 0)""",
        (project_id, next_idx, img_path, width, height)
    )
    total = conn.execute("SELECT COUNT(*) FROM images WHERE project_id = ?", (project_id,)).fetchone()[0]
    conn.execute(
        "UPDATE projects SET total_images = ?, status = 'extracted' WHERE id = ?",
        (total, project_id)
    )
    conn.commit()
    conn.close()

    return {
        "ok": True,
        "project_id": project_id,
        "filename": filename,
        "total_images": total,
        "project_type": "images",
    }


# ─────────────────────────────────────────────────────────────────────────────
# DEMO-MODUS — oeffentliche Kostprobe ohne Anmeldung (nur aktiv bei DEMO_MODE=on)
# ─────────────────────────────────────────────────────────────────────────────
# Diese Endpunkte sind funktional nur in der Demo-Instanz vorhanden (eigener
# Container + eigene Wegwerf-DB, siehe DEMO.md). Bei DEMO_MODE=off liefert der
# Gate-Check 404 — in Production/Staging sind sie damit unsichtbar/inert.
# Sitzungs- und Aufraeum-Logik steckt im Modul demo.py; hier nur die HTTP-Schicht,
# die bewusst die BESTEHENDEN Bausteine (_handle_image_upload, _process_project)
# wiederverwendet — keine eigene Generierungslogik. Limits + Bot-Leitplanke: Schritt 3.

def _require_demo():
    """Schaltet die Demo-Endpunkte ab, wenn die Instanz nicht im Demo-Modus laeuft."""
    if not demo_mod.demo_enabled():
        raise HTTPException(status_code=404, detail="Not found")


def _demo_user_ctx(user_id: int) -> dict:
    """Minimaler Nutzer-Kontext fuer die wiederverwendeten Helfer (brauchen nur id + is_admin)."""
    return {"id": user_id, "is_admin": False}


@app.post("/api/demo/generate")
async def demo_generate(request: Request, file: UploadFile = File(...)):
    """Anonymer Einzelbild-Upload fuer die Demo. Legt bei Bedarf eine fluechtige
    Demo-Sitzung an (signiertes Cookie), speichert das Bild, startet die normale
    Generierungs-Pipeline im Hintergrund und gibt die Projekt-ID zurueck. Das
    Ergebnis wird ueber GET /api/demo/result abgeholt (Polling). Limits: Schritt 3."""
    _require_demo()
    filename = file.filename or "upload"
    ext = os.path.splitext(filename)[1].lower()
    if ext not in IMAGE_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Bitte ein Bild hochladen (JPG, PNG, GIF, WebP …). PDFs gibt es im vollen Werkzeug nach Anmeldung.",
        )
    content = await file.read()
    if len(content) > MAX_UPLOAD_SIZE:
        raise HTTPException(status_code=413, detail=f"Datei zu gross. Maximum: {MAX_UPLOAD_SIZE // (1024*1024)} MB")

    # Limit pruefen (server-seitig, VOR jeder teuren Generierung — nie vom Bot)
    chk = demo_mod.check_generation(request)
    if not chk["allowed"]:
        if chk["global_reached"]:
            raise HTTPException(status_code=429, detail=(
                "Die Demo ist heute stark gefragt — das Tageskontingent ist erreicht. "
                "Bitte morgen wieder probieren oder gleich kostenlos registrieren."))
        raise HTTPException(status_code=429, detail=(
            f"Deine {demo_mod.daily_image_limit()} kostenlosen Analysen für heute sind aufgebraucht. "
            "Für unbegrenzte Bilder, ganze PDFs und den fertig getaggten Export: kostenlos registrieren."))

    # Sitzung aus Cookie ableiten oder neu anlegen
    uid = demo_mod.resolve_session_user_id(request.cookies.get(demo_mod.DEMO_COOKIE_NAME))
    set_cookie = uid is None
    if uid is None:
        uid = demo_mod.create_demo_user(create_user)

    # Datei speichern (eigener Sitzungs-Ordner) — wie im normalen Upload-Pfad
    user_dir = os.path.join(UPLOAD_DIR, str(uid))
    os.makedirs(user_dir, exist_ok=True)
    safe_name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{os.path.basename(filename)}"
    file_path = os.path.join(user_dir, safe_name)
    with open(file_path, "wb") as f:
        f.write(content)

    # BESTEHENDEN Bild-Upload-Handler wiederverwenden (legt Grafik-Projekt + Bild an)
    result = await _handle_image_upload(file_path, filename, _demo_user_ctx(uid), content, ext, None)
    project_id = result["project_id"]

    # Generierung wie im normalen Pfad starten (Hintergrund) — Frontend pollt /result
    conn = get_db()
    conn.execute("UPDATE projects SET status = 'processing' WHERE id = ?", (project_id,))
    conn.commit()
    conn.close()
    asyncio.create_task(_process_project(project_id, uid))
    demo_mod.record_generation(request)  # Zaehler erst nach dem Start hochziehen

    # Secure-Flag korrekt hinter dem Proxy: https in Produktion (Nginx setzt
    # X-Forwarded-Proto), http beim lokalen Dunkel-Tunnel — sonst hielte die
    # Sitzung im Browser ueber den Tunnel nicht.
    is_https = (request.headers.get("x-forwarded-proto", "").lower() == "https"
                or request.url.scheme == "https")
    resp = JSONResponse({"ok": True, "project_id": project_id})
    if set_cookie:
        resp.set_cookie(
            demo_mod.DEMO_COOKIE_NAME, demo_mod.cookie_for(uid),
            max_age=demo_mod.session_ttl_seconds(), httponly=True, samesite="lax", secure=is_https,
        )
    return resp


@app.get("/api/demo/result")
async def demo_result(request: Request):
    """Status + Ergebnis der aktuellen Demo-Sitzung (Ziel des Frontend-Pollings)."""
    _require_demo()
    uid = demo_mod.resolve_session_user_id(request.cookies.get(demo_mod.DEMO_COOKIE_NAME))
    if uid is None:
        raise HTTPException(status_code=404, detail="Keine aktive Demo-Sitzung.")
    demo_mod.touch_session(uid)  # gleitendes TTL: Aktivitaet markieren
    conn = get_db()
    proj = conn.execute(
        "SELECT * FROM projects WHERE user_id = ? ORDER BY id DESC LIMIT 1", (uid,)
    ).fetchone()
    if not proj:
        conn.close()
        return {"ok": True, "status": "leer", "image": None, "limits": demo_mod.remaining_for(request)}
    img = conn.execute(
        "SELECT * FROM images WHERE project_id = ? ORDER BY id DESC LIMIT 1", (proj["id"],)
    ).fetchone()
    conn.close()
    image = None
    if img:
        image = {
            "id": img["id"],
            "status": img["status"],
            "alt_text": _display_alt_text(img),
            "langbeschreibung": img["langbeschreibung"] or "",
            "image_type": img["image_type"] or "",
            "konfidenz": img["konfidenz"] or "",
            "width": img["width"],
            "height": img["height"],
        }
    return {"ok": True, "status": proj["status"], "project_id": proj["id"], "image": image,
            "limits": demo_mod.remaining_for(request)}


@app.delete("/api/demo/session")
async def demo_delete_session(request: Request):
    """Manuelles „Loeschen" durch den Besucher: aktuelle Inhalte sofort weg,
    die Sitzung selbst bleibt bestehen (er kann erneut testen)."""
    _require_demo()
    uid = demo_mod.resolve_session_user_id(request.cookies.get(demo_mod.DEMO_COOKIE_NAME))
    if uid is not None:
        demo_mod.delete_session_content(uid, UPLOAD_DIR, RESULTS_DIR)
    return {"ok": True}


@app.post("/api/demo/chat")
async def demo_chat(request: Request):
    """Anonymer Chat fuer die Demo — derselbe InkluAgent, arbeitet auf dem Bild der
    aktuellen Sitzung. Zwei Demo-Besonderheiten: ein server-seitiges Tageslimit
    (vor dem Modell-Aufruf geprueft) und die thematische Leitplanke, die NUR hier
    an den bestehenden System-Prompt angehaengt wird (siehe demo.DEMO_GUARDRAIL)."""
    _require_demo()
    uid = demo_mod.resolve_session_user_id(request.cookies.get(demo_mod.DEMO_COOKIE_NAME))
    if uid is None:
        raise HTTPException(status_code=404, detail="Keine aktive Demo-Sitzung.")
    data = await request.json()
    message = (data.get("message") or "").strip()
    if not message:
        raise HTTPException(status_code=400, detail="Nachricht darf nicht leer sein")
    if len(message) > 2000:
        raise HTTPException(status_code=400, detail="Nachricht zu lang (max. 2000 Zeichen)")

    chk = demo_mod.check_chat(request)
    if not chk["allowed"]:
        raise HTTPException(status_code=429, detail=(
            f"Du hast die {demo_mod.daily_chat_limit()} kostenlosen Chat-Nachrichten für heute genutzt. "
            "Für unbegrenztes Verfeinern: kostenlos registrieren."))

    conn = get_db()
    proj = conn.execute(
        "SELECT id FROM projects WHERE user_id = ? ORDER BY id DESC LIMIT 1", (uid,)
    ).fetchone()
    conn.close()
    if not proj:
        raise HTTPException(status_code=400, detail="Bitte zuerst ein Bild hochladen, dann hilft der Assistent.")
    project_id = proj["id"]
    demo_mod.touch_session(uid)  # gleitendes TTL

    from inkluagent import storage
    from inkluagent.chat_engine import process_message
    storage.append_message(project_id, "user", message)
    # WICHTIG: system_suffix haengt die Demo-Leitplanke an — der Standard-Prompt
    # des Agenten bleibt unveraendert (Default None im normalen, eingeloggten Pfad).
    result = await asyncio.get_event_loop().run_in_executor(
        None, lambda: process_message(project_id, message, uid, system_suffix=demo_mod.DEMO_GUARDRAIL)
    )
    storage.append_message(
        project_id, "assistant", result["reply"],
        image_refs=result.get("image_refs"), intent=result.get("intent"),
    )
    demo_mod.record_chat(request)
    return {
        "reply": result["reply"],
        "image_refs": result.get("image_refs"),
        "limits": demo_mod.remaining_for(request),
    }


@app.get("/api/demo/limits")
async def demo_limits(request: Request):
    """Aktuelle Rest-Kontingente des Besuchers — auch ohne Sitzung abrufbar
    (fuer die Anzeige 'noch X von 3' beim Laden der Seite)."""
    _require_demo()
    return {"ok": True, "limits": demo_mod.remaining_for(request)}


@app.get("/api/demo/image")
async def demo_image(request: Request):
    """Liefert die Bilddatei der aktuellen Demo-Sitzung (anonym, sitzungs-bezogen)
    — fuer die Wiederherstellung der Vorschau beim Neuladen der Seite."""
    _require_demo()
    uid = demo_mod.resolve_session_user_id(request.cookies.get(demo_mod.DEMO_COOKIE_NAME))
    if uid is None:
        raise HTTPException(status_code=404, detail="Keine aktive Demo-Sitzung.")
    conn = get_db()
    row = conn.execute(
        "SELECT i.image_path FROM images i JOIN projects p ON i.project_id = p.id "
        "WHERE p.user_id = ? ORDER BY i.id DESC LIMIT 1", (uid,)
    ).fetchone()
    conn.close()
    if not row or not os.path.exists(row["image_path"]):
        raise HTTPException(status_code=404, detail="Kein Bild vorhanden.")
    return FileResponse(row["image_path"])


@app.post("/api/demo/save")
async def demo_save(request: Request):
    """Manuelle Aenderungen an Alt-Text/Langbeschreibung speichern (damit der
    Chatbot sie sieht und ein Neuladen sie zeigt). Arbeitet auf dem Bild der Sitzung."""
    _require_demo()
    uid = demo_mod.resolve_session_user_id(request.cookies.get(demo_mod.DEMO_COOKIE_NAME))
    if uid is None:
        raise HTTPException(status_code=404, detail="Keine aktive Demo-Sitzung.")
    data = await request.json()
    alt = (data.get("alt_text") or "").strip()
    lang = (data.get("langbeschreibung") or "").strip()
    conn = get_db()
    row = conn.execute(
        "SELECT i.id FROM images i JOIN projects p ON i.project_id = p.id "
        "WHERE p.user_id = ? ORDER BY i.id DESC LIMIT 1", (uid,)
    ).fetchone()
    if row:
        conn.execute("UPDATE images SET alt_text_edited = ?, langbeschreibung = ? WHERE id = ?",
                     (alt, lang, row["id"]))
        # gleitendes TTL: Aktivitaet markieren (gleiche Verbindung, kein zweiter Connect)
        conn.execute("UPDATE projects SET updated_at = datetime('now') WHERE user_id = ?", (uid,))
        conn.commit()
    conn.close()
    return {"ok": True}


# ─── URL Scanner ─────────────────────────────────────────────

_LINK_PREFIX_PATTERNS = [
    re.compile(r"^(read\s+more|read|more|continue\s+reading)\s*[:\-–—]?\s*", re.IGNORECASE),
    re.compile(r"^(weiterlesen|weiter\s*lesen|mehr\s+lesen|mehr|weiter)\s*[:\-–—]?\s*", re.IGNORECASE),
    re.compile(r"^(lese\s+mehr|lesen)\s*[:\-–—]?\s*", re.IGNORECASE),
]
_LINK_GENERIC_LABELS = {"", "read", "more", "mehr", "weiter", "weiterlesen", "lesen"}


def _clean_link_label(label: str) -> str:
    """Strip generic 'Read more'-prefixes that WordPress screen-reader-text spans
    leak into link text (e.g. '<span class=\"screen-reader-text\">Read</span>:
    DIY Adventskalender' -> 'Read: DIY Adventskalender'). Returns empty string
    if the remainder is purely generic."""
    if not label:
        return ""
    cleaned = label.strip()
    for pat in _LINK_PREFIX_PATTERNS:
        cleaned = pat.sub("", cleaned).strip()
    if cleaned.lower() in _LINK_GENERIC_LABELS or len(cleaned) < 3:
        return ""
    return cleaned


@app.post("/api/scan-url")
async def scan_url(request: Request, user: dict = Depends(get_current_user)):
    """Scan a URL for images and create a project."""
    data = await request.json()
    url = data.get("url", "").strip()
    project_id = data.get("project_id")  # gesetzt = an bestehendes Web-Projekt anhaengen (sonst neues Projekt)

    if not url:
        raise HTTPException(status_code=400, detail="Bitte eine URL eingeben")

    # Schema automatisch ergaenzen: erlaubt Eingaben wie "www.inklutec.de" oder "inklutec.de".
    # Wer bewusst http:// tippt, behaelt es; nur wenn gar kein Schema vorhanden ist, https:// voranstellen.
    if not (url.lower().startswith("http://") or url.lower().startswith("https://")):
        url = "https://" + url

    # Validate URL
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise HTTPException(status_code=400, detail="Nur HTTP/HTTPS-URLs werden unterstuetzt")

    try:
        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
            response = await client.get(url, headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/134.0.0.0 Safari/537.36"
            })
            response.raise_for_status()
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="Zeitueberschreitung beim Laden der Seite")
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=502, detail=f"Seite nicht erreichbar: HTTP {e.response.status_code}")
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Fehler beim Laden der Seite: {str(e)}")

    # Parse HTML
    soup = BeautifulSoup(response.text, "html.parser")
    img_tags = soup.find_all("img")

    if not img_tags:
        raise HTTPException(status_code=404, detail="Keine Bilder auf dieser Seite gefunden")

    # Extract page profile for better context (Seitenprofil)
    from context_engine import extract_page_profile
    page_profile = extract_page_profile(soup)

    # Projekt anlegen ODER an ein bestehendes Web-Projekt anhaengen.
    # Neuer Flow (seit 04.06.2026): Im Dashboard wird zuerst ein leeres "web"-Projekt
    # angelegt; dieses Feld scannt dann eine Seite in genau dieses Projekt, und es koennen
    # nacheinander weitere Seiten angehaengt werden. Ohne project_id (Alt-Flow / direkte API)
    # wird wie bisher ein neues Projekt erzeugt -> voll rueckwaertskompatibel.
    conn = get_db()
    page_title = soup.title.string.strip() if soup.title and soup.title.string else parsed.netloc
    if project_id is not None:
        proj = conn.execute(
            "SELECT id, tool, source_url FROM projects WHERE id = ? AND user_id = ?",
            (project_id, user["id"])
        ).fetchone()
        if not proj:
            conn.close()
            raise HTTPException(status_code=404, detail="Projekt nicht gefunden")
        if proj["tool"] != "web":
            conn.close()
            raise HTTPException(status_code=400, detail="Dieses Projekt ist kein Webseiten-Projekt")
        # Erste gescannte Adresse als Projekt-Quelle merken, falls noch leer (frisch angelegt).
        if not proj["source_url"]:
            conn.execute("UPDATE projects SET source_url = ? WHERE id = ?", (url, project_id))
        conn.execute("UPDATE projects SET status = 'extracting' WHERE id = ?", (project_id,))
        conn.commit()
    else:
        cursor = conn.execute(
            "INSERT INTO projects (user_id, filename, original_path, status, project_type, tool, source_url) VALUES (?, ?, '', 'extracting', 'url', 'web', ?)",
            (user["id"], f"Website: {page_title[:80]}", url)
        )
        project_id = cursor.lastrowid
        conn.commit()

    # Fortlaufende Bildnummer: beim Anhaengen weiterer Seiten ab dem bisherigen Maximum
    # weiterzaehlen. Das haelt image_index eindeutig UND verhindert Dateinamens-Kollisionen
    # (web_<idx>...), weil schon vergebene Indizes nicht erneut benutzt werden.
    base_idx = conn.execute(
        "SELECT COALESCE(MAX(image_index), 0) FROM images WHERE project_id = ?",
        (project_id,)
    ).fetchone()[0]

    img_dir = os.path.join(RESULTS_DIR, str(user["id"]), str(project_id))
    os.makedirs(img_dir, exist_ok=True)

    downloaded = 0
    # Browser-like headers for image downloads: many WordPress sites block requests
    # without Referer header (hotlink protection / bot detection). Fix for 403 errors
    # on sites like dc-tischlermeister.de. Reported by Stephan Raithel, 25.03.2026.
    img_download_headers = {
        "Referer": url,
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/134.0.0.0 Safari/537.36",
        "Accept": "image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8",
    }
    async with httpx.AsyncClient(timeout=15.0, follow_redirects=True, headers=img_download_headers) as client:
        for raw_idx, img_tag in enumerate(img_tags, 1):
            # Beim Anhaengen weiterer Seiten ab dem bisherigen Maximum weiterzaehlen (siehe base_idx).
            idx = base_idx + raw_idx
            # Support lazy-loaded images: check data-src, data-lazy-src first
            src = img_tag.get("data-src") or img_tag.get("data-lazy-src") or img_tag.get("src", "")

            # v2.2.3: Always check srcset for higher-resolution version
            srcset = img_tag.get("srcset") or img_tag.get("data-srcset") or ""
            if srcset:
                # Parse srcset entries and pick the largest by width descriptor
                best_url = ""
                best_width = 0
                for entry in srcset.split(","):
                    parts = entry.strip().split()
                    if len(parts) >= 1:
                        candidate_url = parts[0]
                        w = 0
                        if len(parts) >= 2 and parts[1].endswith("w"):
                            try:
                                w = int(parts[1][:-1])
                            except ValueError:
                                w = 0
                        elif len(parts) >= 2 and parts[1].endswith("x"):
                            try:
                                w = int(float(parts[1][:-1]) * 1000)  # treat 2x as 2000w
                            except ValueError:
                                w = 0
                        if w > best_width:
                            best_width = w
                            best_url = candidate_url
                # Use srcset version if it's larger than the src
                if best_url and best_width > 200:
                    src = best_url

            if not src or src.startswith("data:"):
                continue
            if not src:
                continue

            # Resolve relative URLs
            img_url = urljoin(url, src)

            # Get original alt text
            original_alt = img_tag.get("alt", "")

            # Detect intentionally hidden/decorative images from HTML attributes
            # NOTE: alt="" is NOT treated as hidden – many sites leave alt empty
            # instead of writing proper alt texts. The KI will classify these.
            is_hidden = (
                img_tag.get("aria-hidden") == "true"
                or img_tag.get("role") in ("presentation", "none")
            )
            has_empty_alt = img_tag.has_attr("alt") and img_tag["alt"] == ""

            # Hard bypass: only aria-hidden and role=presentation skip KI
            if is_hidden:
                conn.execute(
                    """INSERT INTO images (project_id, page_number, image_index, image_path, context_text,
                       width, height, xref, original_alt, status, image_type, alt_text)
                       VALUES (?, 1, ?, '', '', 0, 0, 0, ?, 'done', 'dekorativ', '')""",
                    (project_id, idx, original_alt)
                )
                downloaded += 1
                continue

            # Get context: parent text, figcaption, title attribute
            context_parts = []
            if has_empty_alt:
                context_parts.append("[HTML-Hinweis] Bild hat alt=\"\" im Quellcode – prüfe ob wirklich dekorativ oder ob Alt-Text fehlt")
            if img_tag.get("title"):
                context_parts.append(f"[title] {img_tag['title']}")
            # Original alt-text is stored for display but NOT sent to the model
            # to prevent the "discussion bug" (model commenting on existing alt-text)
            # figcaption
            parent_figure = img_tag.find_parent("figure")
            if parent_figure:
                figcaption = parent_figure.find("figcaption")
                if figcaption:
                    context_parts.append(f"[Bildunterschrift] {figcaption.get_text(strip=True)}")
            # Check if image is a link (for linked image alt-text)
            parent_link = img_tag.find_parent("a")
            link_display_text = ""
            if parent_link:
                link_href = parent_link.get("href", "")
                # v2.2.3: Prefer human-readable link text over raw URL
                # v3.7: Sanitize generic "Read more"-prefixes from screen-reader-text spans
                link_label = _clean_link_label(parent_link.get("aria-label", "") or parent_link.get("title", ""))
                # Also check the visible text content of the link (excluding the image alt)
                link_text = parent_link.get_text(strip=True)
                if link_text and link_text.lower() not in {"", "bild", "grafik", "image", "img"}:
                    if not link_label:
                        link_label = _clean_link_label(link_text)[:150]
                if link_label:
                    context_parts.append(f"[Link-Beschriftung] {link_label}")
                    link_display_text = link_label
                if link_href:
                    # Only add raw URL if no readable label available
                    if not link_label:
                        context_parts.append(f"[Link-Ziel] {link_href}")
                        link_display_text = link_href
                    else:
                        # Still store href but label takes priority in context
                        context_parts.append(f"[Link-URL] {link_href}")

            # Improved context search: go beyond direct parent
            # 1. Nearest heading before the image
            prev_heading = img_tag.find_previous(["h1", "h2", "h3", "h4"])
            if prev_heading:
                context_parts.append(f"[Ueberschrift] {prev_heading.get_text(strip=True)[:150]}")

            # 2. Parent article/section (WordPress wrappers)
            content_parent = img_tag.find_parent(["article", "section"]) or img_tag.find_parent("div", class_=True)
            if content_parent and content_parent.name not in ("html", "body", "head"):
                parent_text = content_parent.get_text(strip=True)[:300]
                if parent_text and parent_text != original_alt:
                    context_parts.append(f"[Umgebungstext] {parent_text}")

            # 3. Text after the image (WordPress captions often below)
            next_sib = img_tag.find_next_sibling(["p", "span", "figcaption", "div"])
            if next_sib:
                next_text = next_sib.get_text(strip=True)[:150]
                if next_text:
                    context_parts.append(f"[Text danach] {next_text}")

            # Image-specific context FIRST, page profile as supplement
            all_context = []
            all_context.extend(context_parts)  # Bildspezifisch ZUERST
            if page_profile:
                all_context.append(page_profile)  # Seitenprofil als Ergaenzung
            context_text = "\n".join(all_context) if all_context else ""

            # Download image
            try:
                img_response = await client.get(img_url)
                img_response.raise_for_status()
                img_content = img_response.content

                # Determine extension from content type or URL
                content_type = img_response.headers.get("content-type", "")
                ext = _ext_from_content_type(content_type) or _ext_from_url(img_url)
                if not ext:
                    continue  # Skip unknown formats

                # Convert SVGs to PNG
                if ext == ".svg" or "svg" in content_type:
                    try:
                        import cairosvg
                        png_path = os.path.join(img_dir, f"web_{idx}.png")
                        cairosvg.svg2png(bytestring=img_content, write_to=png_path, output_width=1200)
                        img_path = png_path
                        ext = ".png"
                        img_filename = f"web_{idx}.png"
                    except Exception as e:
                        print(f"SVG-Konvertierung fehlgeschlagen fuer {img_url}: {e}")
                        continue
                else:
                    img_filename = f"web_{idx}{ext}"
                    img_path = os.path.join(img_dir, img_filename)
                    with open(img_path, "wb") as f:
                        f.write(img_content)

                # Get dimensions
                width, height = 0, 0
                try:
                    from PIL import Image as PILImage
                    with PILImage.open(img_path) as pimg:
                        width, height = pimg.size
                except Exception:
                    pass

                # Skip tiny images (likely tracking pixels or icons)
                if width > 0 and height > 0 and (width < 20 or height < 20):
                    os.remove(img_path)
                    continue

                # v2.2.3: If image is a small thumbnail, try to find a larger version
                if width > 0 and height > 0 and width < 250 and height < 250:
                    larger_urls = _try_larger_image_url(img_url)
                    for candidate_url in larger_urls:
                        try:
                            resolved_url = urljoin(url, candidate_url)
                            larger_response = await client.get(resolved_url)
                            if larger_response.status_code == 200:
                                larger_ct = larger_response.headers.get("content-type", "")
                                if "image" in larger_ct:
                                    larger_ext = _ext_from_content_type(larger_ct) or ext
                                    larger_path = os.path.join(img_dir, f"web_{idx}_large{larger_ext}")
                                    with open(larger_path, "wb") as lf:
                                        lf.write(larger_response.content)
                                    try:
                                        with PILImage.open(larger_path) as lpimg:
                                            lw, lh = lpimg.size
                                        if lw > width and lh > height:
                                            # Larger version found! Replace the small one
                                            os.remove(img_path)
                                            new_filename = f"web_{idx}{larger_ext}"
                                            new_path = os.path.join(img_dir, new_filename)
                                            os.rename(larger_path, new_path)
                                            img_path = new_path
                                            img_filename = new_filename
                                            print(f"v2.2.3 Upscale: {width}x{height} -> {lw}x{lh} fuer {img_url[-50:]}")
                                            width, height = lw, lh
                                            break
                                        else:
                                            os.remove(larger_path)
                                    except Exception:
                                        if os.path.exists(larger_path):
                                            os.remove(larger_path)
                        except Exception:
                            pass

                conn.execute(
                    """INSERT INTO images (project_id, page_number, image_index, image_path, context_text,
                       width, height, xref, original_alt)
                       VALUES (?, 1, ?, ?, ?, ?, ?, 0, ?)""",
                    (project_id, idx, img_path, context_text, width, height, original_alt)
                )
                downloaded += 1

            except Exception as e:
                print(f"Fehler beim Download von {img_url}: {e}")
                continue

    # Kumulative Gesamtzahl (beim Anhaengen schon vorhandene Bilder mitzaehlen).
    total = conn.execute("SELECT COUNT(*) FROM images WHERE project_id = ?", (project_id,)).fetchone()[0]
    if downloaded == 0:
        # Beim Anhaengen ein bestehendes Projekt NICHT zerstoeren: hat es schon Bilder,
        # nur freundlich melden und Status zurueck auf 'extracted'. Nur ein wirklich leeres
        # Projekt (Erst-Scan ohne Treffer) wird als Fehler markiert.
        if total > 0:
            conn.execute("UPDATE projects SET status = 'extracted' WHERE id = ?", (project_id,))
            conn.commit()
            conn.close()
            raise HTTPException(status_code=404, detail="Auf dieser Seite wurden keine neuen Bilder gefunden")
        conn.execute("UPDATE projects SET status = 'error' WHERE id = ?", (project_id,))
        conn.commit()
        conn.close()
        raise HTTPException(status_code=404, detail="Keine Bilder konnten heruntergeladen werden")

    conn.execute(
        "UPDATE projects SET status = 'extracted', total_images = ? WHERE id = ?",
        (total, project_id)
    )
    conn.commit()
    conn.close()

    return {
        "ok": True,
        "project_id": project_id,
        "added": downloaded,
        "total_images": total,
        "source_url": url,
        "project_type": "url",
    }



# v2.2.3: Try to find larger versions of thumbnail images via URL rewriting
def _try_larger_image_url(img_url: str) -> list[str]:
    """Generate candidate URLs for larger versions of a thumbnail image.
    Returns a list of alternative URLs to try (most promising first)."""
    candidates = []

    # Pattern 1: BLE/Government CMS - ?__blob=thumbnail -> normal/wide
    if "__blob=thumbnail" in img_url:
        candidates.append(img_url.replace("__blob=thumbnail", "__blob=normal"))
        candidates.append(img_url.replace("__blob=thumbnail", "__blob=wide"))

    # Pattern 2: WordPress - remove size suffix (-150x150, -300x200, etc.)
    import re
    wp_match = re.search(r'-(\d{2,4})x(\d{2,4})\.(\w+)$', img_url)
    if wp_match:
        w, h = int(wp_match.group(1)), int(wp_match.group(2))
        if w <= 300 or h <= 300:
            original = re.sub(r'-\d{2,4}x\d{2,4}\.(\w+)$', r'.\1', img_url)
            candidates.append(original)

    # Pattern 3: Common thumbnail patterns
    replacements = [
        ("_thumb.", "_large."),
        ("_small.", "_large."),
        ("_thumbnail.", "."),
        ("/thumb/", "/"),
        ("/thumbnails/", "/images/"),
        ("/small/", "/large/"),
        ("_s.", "_l."),
        ("-thumb.", "."),
        ("?w=150", "?w=800"),
        ("?width=150", "?width=800"),
        ("&w=150", "&w=800"),
    ]
    for old, new in replacements:
        if old in img_url:
            candidates.append(img_url.replace(old, new))

    # Pattern 4: Shopify CDN
    shopify_match = re.search(r'_(\d+x\d+)\.', img_url)
    if shopify_match and "shopify" in img_url.lower():
        candidates.append(re.sub(r'_\d+x\d+\.', '.', img_url))

    return candidates


def _ext_from_content_type(ct: str) -> str:
    """Map content type to file extension."""
    ct = ct.lower().split(";")[0].strip()
    mapping = {
        "image/jpeg": ".jpg",
        "image/png": ".png",
        "image/gif": ".gif",
        "image/webp": ".webp",
        "image/avif": ".avif",
        "image/heic": ".heic",
        "image/heif": ".heif",
        # SVG excluded – PIL cannot process it, causes UnidentifiedImageError
    }
    return mapping.get(ct, "")


def _ext_from_url(url: str) -> str:
    """Extract file extension from URL path."""
    path = urlparse(url).path.lower()
    for ext in IMAGE_EXTENSIONS:
        if path.endswith(ext):
            return ext
    return ""


def pdf_langbeschreibung_enabled():
    """Schalter (ENV PDF_LANGBESCHREIBUNG): ob bei PDF-Projekten Langbeschreibungen
    erzeugt + angezeigt werden. Standard AUS (Karbe-Wunsch 03.06.2026: bei PDF nicht
    gebraucht, spart den zweiten KI-Pass = Kosten). BILD-Projekte sind NIE betroffen.

    Umschalten OHNE Rebuild: PDF_LANGBESCHREIBUNG=on in .env.staging/.env.prod setzen,
    dann ./compose-staging.sh (bzw. ./compose-prod.sh) up -d  -> Container-Neustart.

    DAUERHAFT ENTFERNEN/AENDERN: diese Funktion + ihre Aufrufstellen (get_project +
    _process_project) sowie die Eintraege in den compose-/env-Dateien anpassen.
    Frontend-Gegenstueck: get_project liefert show_langbeschreibung -> renderImages(showLang)."""
    return os.getenv("PDF_LANGBESCHREIBUNG", "off").strip().lower() in ("1", "true", "on", "yes")


@app.get("/api/projects/{project_id}")
async def get_project(project_id: int, user: dict = Depends(get_current_user)):
    conn = get_db()
    project = conn.execute(
        "SELECT * FROM projects WHERE id = ? AND user_id = ?", (project_id, user["id"])
    ).fetchone()
    if not project:
        conn.close()
        raise HTTPException(status_code=404, detail="Projekt nicht gefunden")

    # Multi-Datei Phase 1 (08.06.2026): Bilder werden weiter geordnet nach
    # Seite/image_index zurueckgegeben — das Frontend gruppiert sie pro Dokument.
    # Reihenfolge: doc_index, page_number, image_index (stabil pro Dokument).
    images = conn.execute(
        """SELECT i.*, (SELECT body FROM messages m WHERE m.image_id = i.id AND m.msg_type = 'review_note' LIMIT 1) AS review_note FROM images i
           LEFT JOIN documents d ON d.id = i.document_id
           WHERE i.project_id = ?
           ORDER BY COALESCE(d.doc_index, 0), i.page_number, i.image_index""",
        (project_id,)
    ).fetchall()
    documents = conn.execute(
        """SELECT id, doc_index, original_filename, display_name, extraction_method, total_images, created_at
           FROM documents WHERE project_id = ? ORDER BY doc_index""",
        (project_id,)
    ).fetchall()
    conn.close()

    # Schalter PDF_LANGBESCHREIBUNG (Standard aus): Langbeschreibung im Frontend nur zeigen,
    # wenn kein PDF-Projekt ODER der Schalter an ist (siehe pdf_langbeschreibung_enabled).
    proj_dict = dict(project)
    _is_pdf = (proj_dict.get("tool") == "pdf" or proj_dict.get("project_type") == "pdf")
    # Review-Status nur zeigen, wenn das Projekt ueberhaupt zur Pruefung freigegeben
    # wurde (Steve 20.06.) -> Solo-Arbeit ohne Einladung bleibt frei von Pruef-Badges.
    _rconn = get_db()
    _in_review = _rconn.execute(
        "SELECT 1 FROM shares WHERE project_id = ? AND status IN ('active','completed') LIMIT 1",
        (project_id,)).fetchone() is not None
    _rconn.close()
    return {
        "project": proj_dict,
        "images": [dict(img) for img in images],
        "documents": [dict(d) for d in documents],
        "show_langbeschreibung": (not _is_pdf) or pdf_langbeschreibung_enabled(),
        "in_review": _in_review,
    }


@app.patch("/api/projects/{project_id}/documents/{document_id}")
async def rename_document(project_id: int, document_id: int, request: Request, user: dict = Depends(get_current_user)):
    """Multi-Datei (08.06.2026): Anzeigename eines Dokuments setzen (Nice-to-have).

    Lasst der Nutzer das Feld leer, faellt die Anzeige im Frontend wieder auf
    den original_filename zurueck (display_name wird auf NULL gesetzt).
    """
    data = await request.json()
    name = (data.get("display_name") or "").strip()
    if len(name) > 200:
        raise HTTPException(status_code=400, detail="Anzeigename darf maximal 200 Zeichen lang sein")
    conn = get_db()
    proj = conn.execute(
        "SELECT id FROM projects WHERE id = ? AND user_id = ?",
        (project_id, user["id"])
    ).fetchone()
    if not proj:
        conn.close()
        raise HTTPException(status_code=404, detail="Projekt nicht gefunden")
    cur = conn.execute(
        "UPDATE documents SET display_name = ? WHERE id = ? AND project_id = ?",
        (name or None, document_id, project_id)
    )
    conn.commit()
    affected = cur.rowcount
    conn.close()
    if not affected:
        raise HTTPException(status_code=404, detail="Dokument nicht gefunden")
    return {"ok": True, "display_name": name or None}


@app.delete("/api/projects/{project_id}/documents/{document_id}")
async def delete_document(project_id: int, document_id: int, user: dict = Depends(get_current_user)):
    """Multi-Datei (10.06.2026): Ein einzelnes Dokument (eine hochgeladene PDF)
    samt seiner Bilder aus einem Projekt entfernen.

    Hintergrund (Michael Karbe, 09.06.2026): Da im Werkzeug stets die KOMPLETTE
    PDF angezeigt wird, loeschen wir bewusst nur auf Dokument-Ebene — einzelne
    Seiten/Bilder einer PDF bleiben unantastbar. (Grafik-/Web-Projekte werden in
    einem Folgeschritt auf Bild-Ebene loeschbar.)

    WICHTIG zur Nummerierung: doc_index wird NICHT neu vergeben. Er ist der
    stabile Ordnerschluessel auf der Platte (results/<user>/<projekt>/doc<N>),
    und neue Uploads leiten ihren Ordner aus MAX(doc_index)+1 ab. Wuerden wir
    umnummerieren, koennten spaetere Uploads in fremde Ordner schreiben. Die fuer
    den Nutzer sichtbare Nummer „Dokument 1, 2, 3" ist reine Anzeige-Position und
    wird im Frontend aus der sortierten Reihenfolge berechnet.

    Mandantensicher: sowohl Projekt- als auch Dokument-Zugriff sind auf den
    eingeloggten Nutzer eingegrenzt (kein IDOR).
    """
    conn = get_db()
    project = conn.execute(
        "SELECT * FROM projects WHERE id = ? AND user_id = ?", (project_id, user["id"])
    ).fetchone()
    if not project:
        conn.close()
        raise HTTPException(status_code=404, detail="Projekt nicht gefunden")

    doc = conn.execute(
        "SELECT * FROM documents WHERE id = ? AND project_id = ?", (document_id, project_id)
    ).fetchone()
    if not doc:
        conn.close()
        raise HTTPException(status_code=404, detail="Dokument nicht gefunden")
    doc = dict(doc)

    # Dateien entfernen: der eigene Bild-/Vorschau-Ordner doc<N> und die
    # hochgeladene Quell-PDF. Best-effort — fehlende Dateien sind kein Fehler.
    doc_dir = os.path.join(RESULTS_DIR, str(user["id"]), str(project_id), f"doc{doc['doc_index']}")
    if os.path.isdir(doc_dir):
        shutil.rmtree(doc_dir, ignore_errors=True)
    src_pdf = doc.get("original_path") or ""
    if src_pdf and os.path.exists(src_pdf):
        try:
            os.remove(src_pdf)
        except OSError:
            pass

    # DB-Zeilen entfernen (Bilder zuerst, dann das Dokument).
    conn.execute("DELETE FROM images WHERE document_id = ? AND project_id = ?", (document_id, project_id))
    conn.execute("DELETE FROM documents WHERE id = ? AND project_id = ?", (document_id, project_id))

    # Projekt-Zaehler aus dem Ist-Stand neu berechnen. processed_images zaehlt
    # weiterhin nur fertige Bilder (status='done'), damit der Fortschritt stimmt.
    remaining_images = conn.execute(
        "SELECT COUNT(*) FROM images WHERE project_id = ?", (project_id,)
    ).fetchone()[0]
    processed = conn.execute(
        "SELECT COUNT(*) FROM images WHERE project_id = ? AND status = 'done'", (project_id,)
    ).fetchone()[0]
    remaining_docs = conn.execute(
        "SELECT COUNT(*) FROM documents WHERE project_id = ?", (project_id,)
    ).fetchone()[0]
    conn.execute(
        "UPDATE projects SET total_images = ?, processed_images = ?, updated_at = datetime('now') WHERE id = ?",
        (remaining_images, processed, project_id)
    )
    conn.commit()
    conn.close()
    return {
        "ok": True,
        "remaining_images": remaining_images,
        "remaining_documents": remaining_docs,
    }


@app.delete("/api/projects/{project_id}")
async def delete_project(project_id: int, user: dict = Depends(get_current_user)):
    conn = get_db()
    project = conn.execute(
        "SELECT * FROM projects WHERE id = ? AND user_id = ?", (project_id, user["id"])
    ).fetchone()
    if not project:
        conn.close()
        raise HTTPException(status_code=404, detail="Projekt nicht gefunden")

    # Delete files
    project_dir = os.path.join(RESULTS_DIR, str(user["id"]), str(project_id))
    if os.path.exists(project_dir):
        shutil.rmtree(project_dir)
    if project["original_path"] and os.path.exists(project["original_path"]):
        os.remove(project["original_path"])

    conn.execute("DELETE FROM images WHERE project_id = ?", (project_id,))
    # Multi-Datei (08.06.2026): Dokument-Zeilen mit aufraeumen.
    conn.execute("DELETE FROM documents WHERE project_id = ?", (project_id,))
    conn.execute("DELETE FROM projects WHERE id = ?", (project_id,))
    conn.commit()
    conn.close()
    return {"ok": True}


@app.get("/api/images/{image_id}/file")
async def get_image_file(image_id: int, user: dict = Depends(get_current_user)):
    conn = get_db()
    img = conn.execute(
        """SELECT i.* FROM images i
           JOIN projects p ON i.project_id = p.id
           WHERE i.id = ? AND p.user_id = ?""",
        (image_id, user["id"])
    ).fetchone()
    conn.close()
    if not img or not os.path.exists(img["image_path"]):
        raise HTTPException(status_code=404, detail="Bild nicht gefunden")
    return FileResponse(img["image_path"])


@app.get("/api/images/{image_id}/page-view")
async def get_image_page_view(image_id: int, user: dict = Depends(get_current_user)):
    """Liefert die Seitenansicht-PNG (ganze PDF-Seite) zur visuellen Kontext-Anzeige."""
    conn = get_db()
    img = conn.execute(
        """SELECT i.page_view_path FROM images i
           JOIN projects p ON i.project_id = p.id
           WHERE i.id = ? AND p.user_id = ?""",
        (image_id, user["id"])
    ).fetchone()
    conn.close()
    if not img or not img["page_view_path"] or not os.path.exists(img["page_view_path"]):
        raise HTTPException(status_code=404, detail="Seitenansicht nicht gefunden")
    return FileResponse(img["page_view_path"])


@app.post("/api/projects/{project_id}/generate")
async def generate_alt_texts(project_id: int, user: dict = Depends(get_current_user)):
    # Check daily limit (admins are exempt)
    if not user.get("is_admin"):
        daily_used = get_daily_image_count(user["id"])
        if daily_used >= DAILY_IMAGE_LIMIT:
            raise HTTPException(status_code=429, detail=f"Tageslimit erreicht ({DAILY_IMAGE_LIMIT} Bilder pro Tag). Das Limit wird um Mitternacht (UTC) zurueckgesetzt.")

    conn = get_db()
    project = conn.execute(
        "SELECT * FROM projects WHERE id = ? AND user_id = ?", (project_id, user["id"])
    ).fetchone()
    if not project:
        conn.close()
        raise HTTPException(status_code=404, detail="Projekt nicht gefunden")

    if project["status"] == "processing":
        conn.close()
        raise HTTPException(status_code=409, detail="Verarbeitung laeuft bereits")

    conn.execute("UPDATE projects SET status = 'processing' WHERE id = ?", (project_id,))
    conn.commit()
    conn.close()

    asyncio.create_task(_process_project(project_id, user["id"]))
    return {"ok": True, "message": "Alt-Text-Generierung gestartet"}


async def _process_project(project_id: int, user_id: int):
    # v2.2.3: Clear duplicate cache for this project
    clear_project_cache()
    conn = get_db()
    images = conn.execute(
        "SELECT * FROM images WHERE project_id = ? AND status = 'pending' ORDER BY page_number, image_index",
        (project_id,)
    ).fetchall()

    # processed_images zaehlt nur echte Erfolge (Steve 08.06.2026 — siehe
    # auch regenerate_image). Fehler werden separat ueber den Bild-Status
    # 'error' sichtbar, blaehen aber nicht den "X von Y Bildern generiert"-
    # Zaehler im UI-Kopf auf.
    processed = conn.execute(
        "SELECT COUNT(*) FROM images WHERE project_id = ? AND status = 'done'",
        (project_id,)
    ).fetchone()[0]
    # PDF-Schalter (pdf_langbeschreibung_enabled): bei PDF + Schalter aus den zweiten KI-Pass
    # (Langbeschreibung) ueberspringen = Kostenersparnis (Karbe 03.06.2026). Bild-Projekte unberuehrt.
    _proj = conn.execute("SELECT tool, project_type, use_context FROM projects WHERE id = ?", (project_id,)).fetchone()
    skip_langbeschreibung = bool(_proj and (_proj["tool"] == "pdf" or _proj["project_type"] == "pdf")) and not pdf_langbeschreibung_enabled()
    # KI-Kontext-Schalter (19.06.2026): bei "aus" bekommt die KI keinen Dokument-
    # Kontext (context=""). Default an (Spalte use_context Default 1). Der Vermerk
    # context_mode wird pro Bild gespeichert (nur Anzeige, NIE exportiert).
    use_context = True if (_proj is None or _proj["use_context"] is None) else bool(_proj["use_context"])
    ctx_mode = "mit" if use_context else "ohne"

    for img in images:
        conn.execute("UPDATE images SET status = 'processing' WHERE id = ?", (img["id"],))
        conn.commit()

        try:
            # v2.2: Pass width, height, original_alt to pipeline
            img_width = img["width"] if img["width"] else 0
            img_height = img["height"] if img["height"] else 0
            img_original_alt = img["original_alt"] if img["original_alt"] else ""

            # KI-Kontext-Schalter: bei "aus" leeren Kontext an die KI geben.
            effective_context = img["context_text"] if use_context else ""

            # First pass: general prompt for type detection + alt-text
            result = await asyncio.get_event_loop().run_in_executor(
                None, generate_alt_text, img["image_path"], effective_context, None,
                img_width, img_height, img_original_alt
            )

            # Second pass: if complex type detected, re-generate with specialized prompt
            # to get langbeschreibung automatically
            from context_engine import is_complex_type
            detected_type = result.get("bildtyp", "")
            langbeschreibung = result.get("langbeschreibung", "")

            if skip_langbeschreibung:
                # PDF + Schalter aus: zweiten Pass ueberspringen, keine Langbeschreibung speichern.
                langbeschreibung = ""
            elif is_complex_type(detected_type) and not langbeschreibung:
                specialized_result = await asyncio.get_event_loop().run_in_executor(
                    None, generate_alt_text, img["image_path"], effective_context, detected_type,
                    img_width, img_height, img_original_alt
                )
                if specialized_result.get("langbeschreibung"):
                    langbeschreibung = specialized_result["langbeschreibung"]
                # Use the specialized alt-text if it's better (has content)
                if specialized_result.get("alt_text") and len(specialized_result["alt_text"]) > 10:
                    result["alt_text"] = specialized_result["alt_text"]
                    result["konfidenz"] = specialized_result.get("konfidenz", result.get("konfidenz", "mittel"))
            conn.execute(
                """UPDATE images SET alt_text = ?, image_type = ?, konfidenz = ?, langbeschreibung = ?,
                   needs_review = ?, pipeline_steps = ?, validation_result = ?, context_mode = ?, status = 'done' WHERE id = ?""",
                (_append_link_reference(result["alt_text"], effective_context or ""), result["bildtyp"], result.get("konfidenz", "mittel"), langbeschreibung,
                 1 if result.get("needs_review") else 0,
                 result.get("pipeline_steps", ""),
                 result.get("validation_result", ""),
                 ctx_mode,
                 img["id"])
            )
            # Nur bei Erfolg den Zaehler hochziehen — Fehler werden separat
            # als status='error' an der Bild-Karte sichtbar.
            processed += 1
        except Exception as e:
            import traceback
            print(f"Fehler bei Bild {img['id']} ({img['image_path']}): {e}")
            print(traceback.format_exc())
            conn.execute("UPDATE images SET status = 'error', alt_text = ? WHERE id = ?",
                         (f"Fehler bei der Analyse: {str(e)[:200]}", img["id"]))

        conn.execute(
            "UPDATE projects SET processed_images = ? WHERE id = ?",
            (processed, project_id)
        )
        conn.commit()

    conn.execute("UPDATE projects SET status = 'done', updated_at = datetime('now') WHERE id = ?", (project_id,))
    conn.commit()
    conn.close()


@app.get("/api/projects/{project_id}/status")
async def get_project_status(project_id: int, user: dict = Depends(get_current_user)):
    conn = get_db()
    project = conn.execute(
        "SELECT * FROM projects WHERE id = ? AND user_id = ?", (project_id, user["id"])
    ).fetchone()
    conn.close()
    if not project:
        raise HTTPException(status_code=404, detail="Projekt nicht gefunden")
    return {
        "status": project["status"],
        "total_images": project["total_images"],
        "processed_images": project["processed_images"],
    }


@app.post("/api/images/{image_id}/alt-text")
async def update_alt_text(image_id: int, request: Request, user: dict = Depends(get_current_user)):
    data = await request.json()
    conn = get_db()
    img = conn.execute(
        """SELECT i.id FROM images i
           JOIN projects p ON i.project_id = p.id
           WHERE i.id = ? AND p.user_id = ?""",
        (image_id, user["id"])
    ).fetchone()
    if not img:
        conn.close()
        raise HTTPException(status_code=404, detail="Bild nicht gefunden")

    conn.execute(
        "UPDATE images SET alt_text_edited = ? WHERE id = ?",
        (data.get("alt_text", ""), image_id)
    )
    # Also update langbeschreibung if provided
    if "langbeschreibung" in data:
        conn.execute(
            "UPDATE images SET langbeschreibung = ? WHERE id = ?",
            (data.get("langbeschreibung", ""), image_id)
        )
    conn.commit()
    conn.close()
    return {"ok": True}


def _display_alt_text(img):
    """Frontend-Fallback in Python: User-Edit > KI-Output > Original aus Quelle.
    Symmetrisch zur Render-Logik in app.html (textarea-value)."""
    return img["alt_text_edited"] or img["alt_text"] or img["original_alt"]


@app.post("/api/images/{image_id}/feedback")
async def submit_feedback(image_id: int, request: Request, user: dict = Depends(get_current_user)):
    """Submit feedback (good/bad) for a generated alt-text."""
    data = await request.json()
    feedback = data.get("feedback", "")
    if feedback not in ("good", "bad"):
        raise HTTPException(status_code=400, detail="Feedback muss 'good' oder 'bad' sein")

    conn = get_db()
    img = conn.execute(
        """SELECT i.*, p.filename as project_name, p.project_type, p.source_url
           FROM images i
           JOIN projects p ON i.project_id = p.id
           WHERE i.id = ? AND p.user_id = ?""",
        (image_id, user["id"])
    ).fetchone()
    if not img:
        conn.close()
        raise HTTPException(status_code=404, detail="Bild nicht gefunden")

    conn.execute("UPDATE images SET feedback = ? WHERE id = ?", (feedback, image_id))
    conn.commit()
    conn.close()

    # Send email notification for ALL feedback (positive + negative)
    alt_text = _display_alt_text(img)
    is_good = feedback == "good"
    color = "#16a34a" if is_good else "#dc2626"
    label = "positiv" if is_good else "negativ"
    emoji = "👍" if is_good else "👎"

    # Build source info based on project type
    project_type = img["project_type"] if "project_type" in img.keys() else "pdf"
    if project_type == "url":
        source_info = f'<p><strong>Quelle:</strong> <a href="{img["source_url"]}">{img["source_url"]}</a></p>'
    elif project_type == "pdf":
        source_info = f'<p><strong>Quelle:</strong> PDF "{img["project_name"]}", Seite {img["page_number"]}, Bild {img["image_index"]}</p>'
    else:
        source_info = f'<p><strong>Quelle:</strong> Einzelbild "{img["project_name"]}"</p>'

    langtext = img["langbeschreibung"] if img["langbeschreibung"] else ""
    lang_section = ""
    if langtext:
        lang_section = f'<p><strong>Langbeschreibung:</strong></p><blockquote style="border-left:3px solid #666;padding-left:1rem;color:#333;">{langtext}</blockquote>'

    email_body = f"""<!DOCTYPE html>
<html lang="de"><head><meta charset="utf-8"></head><body style="font-family:sans-serif;color:#1e293b;max-width:600px;">
<h2 style="color:{color};">{emoji} Alt-Text {label} bewertet</h2>
<p><strong>Benutzer:</strong> {user['email']}</p>
<p><strong>Projekt:</strong> {img['project_name']}</p>
{source_info}
<p><strong>Bildtyp:</strong> {img['image_type']} | <strong>Konfidenz:</strong> {img['konfidenz'] if img['konfidenz'] else 'mittel'}</p>
<p><strong>Bildgröße:</strong> {img['width']}x{img['height']}px</p>
<p><strong>Alt-Text:</strong></p>
<blockquote style="border-left:3px solid {color};padding-left:1rem;color:#333;">{alt_text}</blockquote>
{lang_section}
<p style="color:#64748b;font-size:0.85rem;margin-top:2rem;">InkluDocs Beta-Feedback | Das bewertete Bild ist als Anhang beigefügt.</p>
</body></html>"""

    send_email(
        NOTIFICATION_EMAIL,
        f"InkluDocs: Alt-Text {label} bewertet",
        email_body,
        bcc_admin=False,
        attachment_path=img["image_path"]
    )

    return {"ok": True}


# ─── Regenerate Single Image ────────────────────────────────

@app.post("/api/projects/{project_id}/regenerate/{image_id}")
async def regenerate_image(project_id: int, image_id: int, request: Request, user: dict = Depends(get_current_user)):
    """Regenerate alt-text for a single image with optional specialized prompt."""
    data = await request.json()
    image_type = data.get("image_type")  # Optional: foto, diagramm, karte, etc.
    want_long_desc = data.get("long_description", False)

    conn = get_db()
    img = conn.execute(
        """SELECT i.*, p.use_context AS use_context FROM images i
           JOIN projects p ON i.project_id = p.id
           WHERE i.id = ? AND i.project_id = ? AND p.user_id = ?""",
        (image_id, project_id, user["id"])
    ).fetchone()
    if not img:
        conn.close()
        raise HTTPException(status_code=404, detail="Bild nicht gefunden")

    conn.execute("UPDATE images SET status = 'processing' WHERE id = ?", (image_id,))
    conn.commit()

    try:
        # If long description requested and no explicit type, use the detected type
        effective_type = image_type
        if want_long_desc and not effective_type:
            effective_type = img["image_type"] if img["image_type"] != "unknown" else None

        # v2.2: Pass dimensions and original alt
        regen_width = img["width"] if img["width"] else 0
        regen_height = img["height"] if img["height"] else 0
        regen_original_alt = img["original_alt"] if img["original_alt"] else ""

        # KI-Kontext-Schalter (Projekt-Einstellung): bei "aus" keinen Kontext an die KI.
        _use_context = True if img["use_context"] is None else bool(img["use_context"])
        regen_context = img["context_text"] if _use_context else ""
        regen_ctx_mode = "mit" if _use_context else "ohne"

        # T6 (03.05.2026): Cache evictieren BEVOR die Pipeline laeuft. Ein expliziter
        # Re-Generate-Klick soll alle Cache-Varianten dieses Bildes loeschen (auch andere
        # image_type_override-Varianten), damit kein Mischzustand entsteht. force_regenerate
        # springt zusaetzlich am Cache-Lookup vorbei in generate_alt_text.
        from pdf_processor import _get_image_hash
        from cache import evict_by_content_hash
        try:
            content_hash = _get_image_hash(img["image_path"])
            evicted = evict_by_content_hash(content_hash)
            if evicted:
                print(f"T6 regenerate_image: {evicted} cache-Eintrag(e) fuer {img['image_path']} evictiert")
        except FileNotFoundError:
            pass  # Bild physikalisch weg — Pipeline wird das melden

        result = await asyncio.get_event_loop().run_in_executor(
            None, generate_alt_text, img["image_path"], regen_context, effective_type,
            regen_width, regen_height, regen_original_alt, True  # force_regenerate=True
        )

        langbeschreibung = result.get("langbeschreibung", "")
        conn.execute(
            """UPDATE images SET alt_text = ?, image_type = ?, konfidenz = ?,
               langbeschreibung = ?, alt_text_edited = NULL,
               needs_review = ?, pipeline_steps = ?, validation_result = ?, context_mode = ?, status = 'done' WHERE id = ?""",
            (_append_link_reference(result["alt_text"], regen_context or ""), result["bildtyp"], result.get("konfidenz", "mittel"),
             langbeschreibung,
             1 if result.get("needs_review") else 0,
             result.get("pipeline_steps", ""),
             result.get("validation_result", ""),
             regen_ctx_mode,
             image_id)
        )
        # processed_images zaehlt nur ECHTE Erfolge (Steve 08.06.2026):
        # damit "X von Y Bildern generiert" im UI-Kopf nicht mit Fehlern
        # aufgeblaeht wird. Fehler werden separat ueber die Bild-Karten
        # sichtbar (status='error').
        processed_count = conn.execute(
            "SELECT COUNT(*) FROM images WHERE project_id = ? AND status = 'done'",
            (project_id,)
        ).fetchone()[0]
        conn.execute(
            "UPDATE projects SET processed_images = ? WHERE id = ?",
            (processed_count, project_id)
        )
        conn.commit()
        conn.close()

        return {
            "ok": True,
            "alt_text": result["alt_text"],
            "bildtyp": result["bildtyp"],
            "konfidenz": result.get("konfidenz", "mittel"),
            "langbeschreibung": langbeschreibung,
        }
    except Exception as e:
        # Fehlerpfad: Bild ehrlich als Fehler markieren (Steve 08.06.2026).
        # Vorher wurde status='done' gesetzt — das hat die Bild-Karte als
        # erfolgreich generiert ausgewiesen, obwohl gar nichts produziert
        # wurde. Jetzt 'error' + Pflege von processed_images (nur Erfolge),
        # damit der UI-Zaehler korrekt bleibt.
        conn.execute("UPDATE images SET status = 'error' WHERE id = ?", (image_id,))
        processed_count = conn.execute(
            "SELECT COUNT(*) FROM images WHERE project_id = ? AND status = 'done'",
            (project_id,)
        ).fetchone()[0]
        conn.execute(
            "UPDATE projects SET processed_images = ? WHERE id = ?",
            (processed_count, project_id)
        )
        conn.commit()
        conn.close()
        raise HTTPException(status_code=500, detail=f"Fehler bei der Neugenerierung: {str(e)}")


# ─── Export Routes ───────────────────────────────────────────

# ─── Multi-Datei Export-Helpers (08.06.2026) ─────────────────
#
# Phase 1 dreht jedes "ein PDF pro Projekt" in "mehrere PDFs pro Projekt".
# Damit der bestehende Export-Code nicht doppelt geschrieben werden muss,
# kapseln wir die Datei-Erzeugung pro Dokument in kleinen Helfern und rufen
# diese aus den oeffentlichen Endpunkten heraus. Auswahl-Logik:
#   - kein document_id Parameter   -> alle Dokumente, gepackt als ZIP
#   - ein document_id Parameter    -> nur dieses Dokument, direkter Download
# Der ZIP-Name kann optional ueber den 'filename'-Body-Parameter gesetzt
# werden (ohne Erweiterung), genauso wie bei Einzelexports.

import zipfile
from typing import Optional


def _safe_filename_component(name: str, fallback: str = "datei") -> str:
    """Macht aus einem beliebigen Anzeigenamen einen Datei-System-tauglichen
    Bestandteil (keine Slashes, keine Steuerzeichen, max. 120 Zeichen).
    Wir bleiben bewusst konservativ, damit ZIP-Innennamen auch unter Windows
    sauber bleiben."""
    if not name:
        return fallback
    s = re.sub(r"[\\/\x00-\x1f<>:\"|?*]", "_", str(name)).strip(" .")
    return s[:120] or fallback


async def _read_export_options(request: Optional[Request]) -> tuple[Optional[int], Optional[str]]:
    """Liest optionale Export-Parameter (document_id + filename) aus dem Request-Body.
    Beide Felder sind optional; fehlt der Body komplett, faellt alles auf None zurueck."""
    if request is None:
        return None, None
    try:
        body = await request.json()
    except Exception:
        return None, None
    doc_id = body.get("document_id") if isinstance(body, dict) else None
    if doc_id is not None:
        try:
            doc_id = int(doc_id)
        except (TypeError, ValueError):
            doc_id = None
    name = (body.get("filename") if isinstance(body, dict) else None) or None
    if name:
        name = str(name).strip()
        name = re.sub(r"\.[A-Za-z0-9]{1,5}$", "", name)
        name = _safe_filename_component(name)
    return doc_id, name


def _doc_label(doc: dict) -> str:
    """Anzeigename eines Dokuments fuer Dateinamen (display_name -> original_filename ohne .pdf)."""
    raw = (doc.get("display_name") or "").strip() or (doc.get("original_filename") or "").strip()
    raw = re.sub(r"\.pdf$", "", raw, flags=re.IGNORECASE)
    return _safe_filename_component(raw, f"dokument_{doc.get('doc_index', 1)}")


def _load_pdf_export_units(project: dict, user_id: int, document_id: Optional[int]) -> list[dict]:
    """Lieferst Liste aus dicts mit doc-Metadaten + alt-Texten + image-Metadaten
    fuer jedes Dokument, das exportiert werden soll. Phase 1 verarbeitet nur
    PDF-Werkzeuge — andere Tools liefern eine einzelne synthetische Einheit
    (alles im Projekt zusammen)."""
    conn = get_db()
    docs = conn.execute(
        """SELECT * FROM documents WHERE project_id = ? ORDER BY doc_index""",
        (project["id"],)
    ).fetchall()
    if document_id is not None:
        docs = [d for d in docs if d["id"] == document_id]
        if not docs:
            conn.close()
            raise HTTPException(status_code=404, detail="Dokument nicht gefunden")

    units = []
    for d in docs:
        imgs = conn.execute(
            "SELECT * FROM images WHERE document_id = ? ORDER BY page_number, image_index",
            (d["id"],)
        ).fetchall()
        units.append({"doc": dict(d), "images": [dict(i) for i in imgs]})
    conn.close()

    # Fallback fuer den Uebergang: ein PDF-Projekt OHNE documents-Eintraege
    # (z.B. Race-Bedingung beim Erst-Migrate) wird wie ein einzelnes Dokument
    # behandelt, das auf project.filename/original_path zeigt.
    if not units:
        conn = get_db()
        imgs = conn.execute(
            "SELECT * FROM images WHERE project_id = ? ORDER BY page_number, image_index",
            (project["id"],)
        ).fetchall()
        conn.close()
        units.append({
            "doc": {
                "id": None,
                "doc_index": 1,
                "original_filename": project.get("filename") or "dokument.pdf",
                "display_name": None,
                "original_path": project.get("original_path") or "",
                "extraction_method": project.get("extraction_method") or "fitz",
            },
            "images": [dict(i) for i in imgs],
        })
    return units


def _exportable_alt_text(img) -> Optional[str]:
    """Alt-Text eines Bildes fuer den PDF-Export, oder None wenn das Bild
    NICHT exportiert werden soll.

    Regel (12.06.2026): Nur Bilder mit echtem Text (oder der expliziten
    Markierung "dekorativ") kommen in den Export. Unbearbeitete Bilder
    (leerer Text, z.B. status='pending') werden uebersprungen — sie bleiben
    in der PDF exakt wie im Original, inklusive eines evtl. vorhandenen
    Original-Alt-Textes. Vorher schrieb der Export leere /Alt-Eintraege in
    die Datei (Befund 12.06.2026: Export eines teilbearbeiteten Projekts
    erzeugte 6 leere Alt-Texte, die Pruefwerkzeuge als Fehler werten)."""
    alt_text = _display_alt_text(img)
    if alt_text is None:
        return None
    if alt_text == "dekorativ" or alt_text.strip():
        return alt_text
    return None


def _build_pdf_for_document(unit: dict, output_dir: str,
                            custom_title: Optional[str] = None) -> tuple[str, dict]:
    """Erzeugt die exportierte PDF fuer EIN Dokument (alle Alt-Texte
    eingebettet). Gibt (output_path, header_info) zurueck. Header_info
    enthaelt die gleichen Metriken wie der Single-Export davor.
    custom_title: vom Nutzer beim Export vergebener Name — wird als
    PDF-Dokumenttitel uebernommen (siehe finalize_export_pdf)."""
    doc = unit["doc"]
    images = unit["images"]
    extraction_method = doc.get("extraction_method") or "fitz"

    alt_texts = {}
    alt_texts_by_lfnr = {}
    image_metadata = []

    for img in images:
        alt_text = _exportable_alt_text(img)
        if alt_text is not None and img.get("xref"):
            alt_texts[img["xref"]] = alt_text
        if alt_text is not None and img.get("image_index"):
            alt_texts_by_lfnr[int(img["image_index"])] = alt_text
        bbox = None
        if img.get("bbox_x0") is not None:
            bbox = (img["bbox_x0"], img["bbox_y0"], img["bbox_x1"], img["bbox_y1"])
        image_metadata.append({
            "xref": img.get("xref"),
            "page_number": img.get("page_number"),
            "is_vector": bool(img.get("is_vector")) if img.get("is_vector") is not None else False,
            "bbox": bbox,
            "alt_text": alt_text,
            "image_path": img.get("image_path"),
        })

    os.makedirs(output_dir, exist_ok=True)
    base = doc.get("original_filename") or f"dokument_{doc.get('doc_index', 1)}.pdf"
    output_path = os.path.join(output_dir, f"inkludocs_{_safe_filename_component(base, base)}")
    info: dict = {}

    if extraction_method == "pdfix":
        try:
            import pdfix_roundtrip as _pdfix
            count = _pdfix.import_alt_texts_pdfix(
                doc["original_path"], output_path,
                alt_texts_by_lfnr, work_dir=output_dir)
            info["method"] = "pdfix"
            info["tagged"] = count
            info["total"] = len(alt_texts_by_lfnr)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"PDFix-Export fehlgeschlagen: {str(e)}")
    else:
        try:
            from pdf_export import write_alt_texts_to_pdf
            result = write_alt_texts_to_pdf(doc["original_path"], output_path, alt_texts, image_metadata)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Export fehlgeschlagen: {str(e)}")
        if isinstance(result, dict):
            info["method"] = "fitz"
            info["tagged"] = result.get("tagged_count", 0)
            info["total"] = len(alt_texts)
            warnings = result.get("warnings") or []
            if warnings:
                info["warnings"] = warnings

    # Gemeinsamer Abschluss fuer BEIDE Pfade (12.06.2026): Dokumentsprache,
    # Dokumenttitel (WCAG 3.1.1 / 2.4.2) und verwaiste /Alt-Altlasten der
    # Quell-PDF entfernen. Titel-Prioritaet: Export-Name des Nutzers >
    # Umbenennung in InkluDocs (display_name) > vorhandener Titel der
    # Quell-PDF > Dateiname ohne Endung. Der Dokument-Inhalt bleibt unberuehrt.
    explicit_title = (custom_title or "").strip() or (doc.get("display_name") or "").strip() or None
    fallback_title = re.sub(r"\.pdf$", "", (doc.get("original_filename") or "").strip(),
                            flags=re.IGNORECASE) or None
    try:
        from pdf_export import finalize_export_pdf
        info["a11y"] = finalize_export_pdf(output_path, title=explicit_title,
                                           fallback_title=fallback_title)
    except Exception as e:
        # Der Abschluss-Schritt darf den Export nicht scheitern lassen:
        # Die Alt-Texte sind zu diesem Zeitpunkt bereits korrekt gesetzt,
        # Sprache/Titel/Aufraeumen sind Zusatznutzen. Fehler aber loggen.
        print(f"WARNUNG: finalize_export_pdf fehlgeschlagen fuer {output_path}: {e}")

    return output_path, info


@app.post("/api/projects/{project_id}/export")
async def export_pdf(project_id: int, request: Request, user: dict = Depends(get_current_user)):
    """PDF-Export. Multi-Datei (08.06.2026):
    - ohne document_id im Body: alle Dokumente als ZIP
    - mit document_id: nur dieses Dokument, direkter Download
    """
    document_id, custom_name = await _read_export_options(request)
    conn = get_db()
    project = conn.execute(
        "SELECT * FROM projects WHERE id = ? AND user_id = ?", (project_id, user["id"])
    ).fetchone()
    if not project:
        conn.close()
        raise HTTPException(status_code=404, detail="Projekt nicht gefunden")
    project_type = project["project_type"] if "project_type" in project.keys() else "pdf"
    if project_type != "pdf":
        conn.close()
        raise HTTPException(status_code=400, detail="PDF-Export ist nur fuer PDF-Projekte verfuegbar")
    project = dict(project)
    conn.close()

    units = _load_pdf_export_units(project, user["id"], document_id)
    output_dir = os.path.join(RESULTS_DIR, str(user["id"]), str(project["id"]), "_export")
    os.makedirs(output_dir, exist_ok=True)

    if document_id is not None or len(units) == 1:
        # Einzelne Datei zurueckgeben (direkter Download, kein ZIP).
        unit = units[0]
        output_path, info = _build_pdf_for_document(unit, output_dir,
                                                    custom_title=custom_name)
        headers = {}
        if info.get("method"):
            headers["X-Export-Method"] = str(info["method"])
        if "tagged" in info:
            headers["X-Export-Tagged"] = str(info["tagged"])
        if "total" in info:
            headers["X-Export-Total"] = str(info["total"])
        if info.get("warnings"):
            headers["X-Export-Warnings"] = json.dumps(info["warnings"], ensure_ascii=False)
        download_base = custom_name or _doc_label(unit["doc"])
        return FileResponse(
            output_path,
            filename=f"inkludocs_{download_base}.pdf",
            media_type="application/pdf",
            headers=headers,
        )

    # Alle Dokumente -> ZIP.
    zip_base = custom_name or _safe_filename_component(project.get("name") or project.get("filename") or "projekt")
    zip_path = os.path.join(output_dir, f"{zip_base}_alle_pdfs.zip")
    aggregated_warnings: list[str] = []
    total_tagged = 0
    total_images = 0
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        # Nummerierung nach Anzeige-Position (1..N), passend zur „Dokument N"-
        # Anzeige im Frontend — NICHT nach doc_index, der nach Loeschungen Luecken
        # haben kann.
        for pos, unit in enumerate(units, start=1):
            out_path, info = _build_pdf_for_document(unit, output_dir)
            inner = f"{pos:02d}_{_doc_label(unit['doc'])}.pdf"
            zf.write(out_path, arcname=inner)
            total_tagged += int(info.get("tagged", 0) or 0)
            total_images += int(info.get("total", 0) or 0)
            for w in info.get("warnings", []) or []:
                aggregated_warnings.append(f"[{inner}] {w}")
    headers = {
        "X-Export-Tagged": str(total_tagged),
        "X-Export-Total": str(total_images),
    }
    if aggregated_warnings:
        headers["X-Export-Warnings"] = json.dumps(aggregated_warnings, ensure_ascii=False)
    return FileResponse(
        zip_path,
        filename=f"{zip_base}_alle_pdfs.zip",
        media_type="application/zip",
        headers=headers,
    )


# ─── Multi-Datei Export: JSON / CSV / XLSX (08.06.2026) ────────
#
# Diese drei Formate koennen INHALTLICH wahlweise nur ein Dokument
# oder alle gemeinsam darstellen. Wir gehen den simplen, fuer Nutzer
# erwartbaren Weg:
#   - document_id im Body  -> nur dieses Dokument, eine Datei
#   - sonst                -> alle Dokumente als ZIP (eine Datei pro Doc)
# Web- und Grafik-Projekte (kein documents-Eintrag) bleiben ein einzelnes
# logisches Dokument und liefern unveraendert eine flache Einzeldatei.

def _load_export_units_for_table(project: dict, document_id: Optional[int]) -> list[dict]:
    """Wie _load_pdf_export_units, aber NICHT auf project_type='pdf' beschraenkt.
    Web/Grafik-Projekte haben kein documents-Eintrag -> Fallback auf ein
    einziges synthetisches Dokument (alle Bilder zusammen)."""
    conn = get_db()
    docs = conn.execute(
        "SELECT * FROM documents WHERE project_id = ? ORDER BY doc_index",
        (project["id"],)
    ).fetchall()
    if docs:
        if document_id is not None:
            docs = [d for d in docs if d["id"] == document_id]
            if not docs:
                conn.close()
                raise HTTPException(status_code=404, detail="Dokument nicht gefunden")
        units = []
        for d in docs:
            imgs = conn.execute(
                "SELECT * FROM images WHERE document_id = ? ORDER BY page_number, image_index",
                (d["id"],)
            ).fetchall()
            units.append({"doc": dict(d), "images": [dict(i) for i in imgs]})
        conn.close()
        return units
    # Kein Dokument-Eintrag (Web/Grafik oder noch nicht migriert).
    imgs = conn.execute(
        "SELECT * FROM images WHERE project_id = ? ORDER BY page_number, image_index",
        (project["id"],)
    ).fetchall()
    conn.close()
    return [{
        "doc": {
            "id": None,
            "doc_index": 1,
            "original_filename": project.get("filename") or "export",
            "display_name": None,
        },
        "images": [dict(i) for i in imgs],
    }]


def _build_json_bytes(unit: dict, project_name: str) -> bytes:
    export_data = {
        "projekt": project_name,
        "dokument": _doc_label(unit["doc"]),
        "bilder": [],
    }
    for img in unit["images"]:
        alt_text = _display_alt_text(img)
        entry = {"alt_text": alt_text or ""}
        lang = img.get("langbeschreibung") or ""
        if lang:
            entry["langbeschreibung"] = lang
        export_data["bilder"].append(entry)
    return json.dumps(export_data, ensure_ascii=False, indent=2).encode("utf-8")


def _build_csv_bytes(unit: dict) -> bytes:
    output = io.StringIO()
    writer = csv.writer(output, delimiter=";")
    writer.writerow(["Alt-Text", "Langbeschreibung"])
    for img in unit["images"]:
        alt_text = _display_alt_text(img)
        writer.writerow([alt_text or "", img.get("langbeschreibung") or ""])
    return output.getvalue().encode("utf-8-sig")


def _build_xlsx_bytes(unit: dict) -> bytes:
    from openpyxl import Workbook
    from openpyxl.drawing.image import Image as XlImage
    from openpyxl.styles import Font, Alignment

    wb = Workbook()
    ws = wb.active
    ws.title = "Alt-Texte"
    ws["A1"] = "Bild"
    ws["B1"] = "Alt-Text"
    ws["C1"] = "Langbeschreibung"
    for cell in [ws["A1"], ws["B1"], ws["C1"]]:
        cell.font = Font(bold=True, size=12)
    ws.column_dimensions["A"].width = 25
    ws.column_dimensions["B"].width = 60
    ws.column_dimensions["C"].width = 60

    for i, img in enumerate(unit["images"]):
        row = i + 2
        alt_text = _display_alt_text(img)
        langbeschreibung = img.get("langbeschreibung") or ""
        img_path = img.get("image_path") or ""
        img_filename = os.path.basename(img_path) if img_path else "unbekannt"
        ws[f"A{row}"] = img_filename
        ws[f"A{row}"].alignment = Alignment(vertical="top")
        if img_path and os.path.exists(img_path):
            try:
                export_img_path = img_path
                if img_path.lower().endswith((".webp", ".avif", ".heic", ".heif")):
                    from PIL import Image as PILImage
                    pil_img = PILImage.open(img_path)
                    if pil_img.mode not in ("RGB", "L"):
                        pil_img = pil_img.convert("RGB")
                    png_path = img_path.rsplit(".", 1)[0] + "_export.png"
                    pil_img.save(png_path, format="PNG")
                    export_img_path = png_path
                xl_img = XlImage(export_img_path)
                max_w = 150
                max_h = 120
                ratio = min(max_w / xl_img.width, max_h / xl_img.height, 1.0)
                xl_img.width = int(xl_img.width * ratio)
                xl_img.height = int(xl_img.height * ratio)
                ws.row_dimensions[row].height = max(xl_img.height * 0.75, 60)
                ws.add_image(xl_img, f"A{row}")
            except Exception:
                pass
        ws[f"B{row}"] = alt_text or ""
        ws[f"B{row}"].alignment = Alignment(wrap_text=True, vertical="top")
        ws[f"C{row}"] = langbeschreibung
        ws[f"C{row}"].alignment = Alignment(wrap_text=True, vertical="top")

    buf = io.BytesIO()
    wb.save(buf)
    buf.seek(0)
    return buf.getvalue()


async def _table_export_dispatch(project_id: int, request: Request, user: dict, fmt: str):
    """Gemeinsamer Code fuer JSON/CSV/XLSX-Export."""
    document_id, custom_name = await _read_export_options(request)
    conn = get_db()
    project = conn.execute(
        "SELECT * FROM projects WHERE id = ? AND user_id = ?", (project_id, user["id"])
    ).fetchone()
    if not project:
        conn.close()
        raise HTTPException(status_code=404, detail="Projekt nicht gefunden")
    project = dict(project)
    conn.close()

    units = _load_export_units_for_table(project, document_id)
    project_name = project.get("name") or project.get("filename") or "Projekt"

    if fmt == "json":
        media = "application/json"
        suffix = ".json"
        single = lambda u: _build_json_bytes(u, project_name)
    elif fmt == "csv":
        media = "text/csv"
        suffix = ".csv"
        single = lambda u: _build_csv_bytes(u)
    elif fmt == "xlsx":
        media = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        suffix = ".xlsx"
        single = lambda u: _build_xlsx_bytes(u)
    else:
        raise HTTPException(status_code=400, detail="Unbekanntes Format")

    if document_id is not None or len(units) == 1:
        unit = units[0]
        data = single(unit)
        base = custom_name or _doc_label(unit["doc"])
        return StreamingResponse(
            io.BytesIO(data),
            media_type=media,
            headers={"Content-Disposition": f'attachment; filename="inkludocs_{base}{suffix}"'}
        )

    # Mehrere Dokumente -> ZIP
    zip_base = custom_name or _safe_filename_component(project_name)
    output_dir = os.path.join(RESULTS_DIR, str(user["id"]), str(project["id"]), "_export")
    os.makedirs(output_dir, exist_ok=True)
    zip_path = os.path.join(output_dir, f"{zip_base}_alle_{fmt}.zip")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        # Nummerierung nach Anzeige-Position (1..N), siehe PDF-Export oben.
        for pos, unit in enumerate(units, start=1):
            inner = f"{pos:02d}_{_doc_label(unit['doc'])}{suffix}"
            zf.writestr(inner, single(unit))
    return FileResponse(
        zip_path,
        filename=f"{zip_base}_alle_{fmt}.zip",
        media_type="application/zip",
    )


@app.post("/api/projects/{project_id}/export/json")
async def export_json(project_id: int, request: Request, user: dict = Depends(get_current_user)):
    """Multi-Datei: ohne document_id alle Dokumente als ZIP, sonst Einzeldatei."""
    return await _table_export_dispatch(project_id, request, user, "json")


@app.post("/api/projects/{project_id}/export/csv")
async def export_csv(project_id: int, request: Request, user: dict = Depends(get_current_user)):
    """Multi-Datei: ohne document_id alle Dokumente als ZIP, sonst Einzeldatei."""
    return await _table_export_dispatch(project_id, request, user, "csv")


@app.post("/api/projects/{project_id}/export/xlsx")
async def export_xlsx(project_id: int, request: Request, user: dict = Depends(get_current_user)):
    """Multi-Datei: ohne document_id alle Dokumente als ZIP, sonst Einzeldatei."""
    return await _table_export_dispatch(project_id, request, user, "xlsx")


# ─── Public API ──────────────────────────────────────────────

@app.post("/api/v1/alt-text")
async def api_generate_alt_text(request: Request):
    """Public API endpoint for alt-text generation. Requires X-API-Key header.
    Accepts multipart/form-data (file upload) or application/json (base64 image)."""
    import base64 as b64module

    api_user = get_api_user(request)
    api_key_id = api_user["api_key_id"]

    # Daily limit check
    daily_used = get_daily_api_count(api_user["id"])
    daily_remaining = max(0, DAILY_IMAGE_LIMIT - daily_used)
    if daily_used >= DAILY_IMAGE_LIMIT:
        raise HTTPException(
            status_code=429,
            detail=f"Tageslimit erreicht ({DAILY_IMAGE_LIMIT} Bilder pro Tag). Das Limit wird um Mitternacht (UTC) zurueckgesetzt.",
            headers={"X-RateLimit-Limit": str(DAILY_IMAGE_LIMIT), "X-RateLimit-Remaining": "0"},
        )

    # Rate limiting
    rate_info = check_api_rate_limit(api_key_id)

    content_type = request.headers.get("content-type", "")
    context_text = ""
    language = "de"
    image_type = None
    content = None
    ext = ".jpg"

    if "application/json" in content_type:
        # JSON mode: base64-encoded image
        try:
            data = await request.json()
        except Exception:
            raise HTTPException(status_code=400, detail="Ungueltiges JSON")

        image_b64 = data.get("image_base64", "")
        if not image_b64:
            log_api_usage(api_key_id, api_user["id"], success=False,
                          error_message="Kein Bild (Base64)")
            raise HTTPException(status_code=400, detail="Feld 'image_base64' fehlt oder ist leer")

        context_text = data.get("context", "")
        language = data.get("language", "de")
        image_type = data.get("image_type")

        # Strip data URI prefix if present (e.g. "data:image/png;base64,...")
        if "," in image_b64 and image_b64.startswith("data:"):
            mime_part = image_b64.split(",")[0]  # "data:image/png;base64"
            image_b64 = image_b64.split(",", 1)[1]
            # Detect extension from MIME
            if "png" in mime_part: ext = ".png"
            elif "gif" in mime_part: ext = ".gif"
            elif "webp" in mime_part: ext = ".webp"
            else: ext = ".jpg"

        try:
            content = b64module.b64decode(image_b64)
        except Exception:
            log_api_usage(api_key_id, api_user["id"], success=False,
                          error_message="Base64 ungueltig")
            raise HTTPException(status_code=400, detail="Base64-Daten konnten nicht dekodiert werden")

    else:
        # Multipart mode: file upload
        form = await request.form()
        file = form.get("file") or form.get("image")
        if not file:
            log_api_usage(api_key_id, api_user["id"], success=False,
                          error_message="Kein Bild mitgeschickt")
            raise HTTPException(status_code=400, detail="Bitte eine Bilddatei als 'file' oder 'image' hochladen")

        context_text = form.get("context", "")
        language = form.get("language", "de")
        image_type = form.get("image_type")

        filename = file.filename or "image.jpg"
        ext = os.path.splitext(filename)[1].lower()
        if ext not in IMAGE_EXTENSIONS:
            log_api_usage(api_key_id, api_user["id"], success=False,
                          error_message=f"Ungueltiges Format: {ext}")
            raise HTTPException(
                status_code=400,
                detail="Nur Bilddateien erlaubt (JPG, PNG, GIF, WebP, BMP, TIFF, HEIC)"
            )
        content = await file.read()

    image_size = len(content)
    if image_size > 10 * 1024 * 1024:  # 10 MB limit for API
        log_api_usage(api_key_id, api_user["id"], image_size_bytes=image_size,
                      success=False, error_message="Bild zu gross")
        raise HTTPException(status_code=413, detail="Bild zu gross. Maximum: 10 MB")

    # Save temporarily
    tmp_dir = os.path.join(UPLOAD_DIR, "api_tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    tmp_path = os.path.join(tmp_dir, f"{secrets.token_hex(8)}{ext}")
    with open(tmp_path, "wb") as f:
        f.write(content)

    start_time = time.time()
    try:
        # v2.2: Get image dimensions for thumbnail guard
        try:
            from PIL import Image as PILImage
            with PILImage.open(tmp_path) as _pil_img:
                api_img_width, api_img_height = _pil_img.size
        except Exception:
            api_img_width, api_img_height = 0, 0

        result = await asyncio.get_event_loop().run_in_executor(
            None, generate_alt_text_for_image, tmp_path, context_text, image_type,
            api_img_width, api_img_height, ""
        )
        processing_time_ms = int((time.time() - start_time) * 1000)
        model_used = result.get("model_used", "")

        # Log successful usage
        log_api_usage(api_key_id, api_user["id"],
                      processing_time_ms=processing_time_ms,
                      model_used=model_used,
                      image_size_bytes=image_size, success=True)

        # result_id generieren und Ergebnis speichern
        result_id = secrets.token_urlsafe(16)
        create_api_result(
            result_id=result_id,
            user_id=api_user["id"],
            api_key_id=api_key_id,
            alt_text=result.get("alt_text", ""),
            langbeschreibung=result.get("langbeschreibung", ""),
            bildtyp=result.get("bildtyp", "unbekannt"),
            konfidenz=result.get("konfidenz", "mittel"),
            model_used=model_used,
            processing_time_ms=processing_time_ms,
            language=language,
            context_text=context_text,
        )

        response_data = {
            "result_id": result_id,
            "alt_text": result.get("alt_text", ""),
            "langbeschreibung": result.get("langbeschreibung", ""),
            "bildtyp": result.get("bildtyp", "unbekannt"),
            "konfidenz": result.get("konfidenz", "mittel"),
            "processing_time_ms": processing_time_ms,
        }
        return JSONResponse(
            content=response_data,
            headers={
                "X-RateLimit-Remaining-Minute": str(rate_info["minute_remaining"]),
                "X-RateLimit-Remaining-Day": str(rate_info["day_remaining"]),
                "X-DailyLimit-Limit": str(DAILY_IMAGE_LIMIT),
                "X-DailyLimit-Used": str(daily_used + 1),
                "X-DailyLimit-Remaining": str(max(0, daily_remaining - 1)),
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        processing_time_ms = int((time.time() - start_time) * 1000)
        log_api_usage(api_key_id, api_user["id"],
                      processing_time_ms=processing_time_ms,
                      image_size_bytes=image_size, success=False,
                      error_message=str(e)[:500])
        raise HTTPException(status_code=500, detail="Interner Fehler bei der Alt-Text-Generierung")
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass


@app.get("/api/v1/alt-text/{result_id}")
async def api_get_alt_text(result_id: str, request: Request):
    """Ruft ein gespeichertes API-Ergebnis ab. Erfordert X-API-Key Header."""
    api_user = get_api_user(request)

    result = get_api_result(result_id, api_user["id"])
    if not result:
        raise HTTPException(status_code=404, detail="Ergebnis nicht gefunden")

    return JSONResponse(content={
        "result_id": result["id"],
        "alt_text": result["alt_text"],
        "langbeschreibung": result["langbeschreibung"],
        "bildtyp": result["bildtyp"],
        "konfidenz": result["konfidenz"],
        "created_at": result["created_at"],
        "updated_at": result["updated_at"],
    })


@app.patch("/api/v1/alt-text/{result_id}")
async def api_update_alt_text(result_id: str, request: Request):
    """Aktualisiert den Alt-Text und/oder die Langbeschreibung eines API-Ergebnisses.
    Erfordert X-API-Key Header. Nur der Besitzer kann sein Ergebnis aendern."""
    api_user = get_api_user(request)

    try:
        data = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Ungueltiges JSON")

    alt_text = data.get("alt_text")
    langbeschreibung = data.get("langbeschreibung")

    if alt_text is None and langbeschreibung is None:
        raise HTTPException(
            status_code=400,
            detail="Mindestens 'alt_text' oder 'langbeschreibung' muss angegeben werden"
        )

    if alt_text is not None and not isinstance(alt_text, str):
        raise HTTPException(status_code=400, detail="'alt_text' muss ein String sein")
    if langbeschreibung is not None and not isinstance(langbeschreibung, str):
        raise HTTPException(status_code=400, detail="'langbeschreibung' muss ein String sein")

    success = update_api_result(
        result_id=result_id,
        user_id=api_user["id"],
        alt_text=alt_text,
        langbeschreibung=langbeschreibung,
    )

    if not success:
        raise HTTPException(status_code=404, detail="Ergebnis nicht gefunden")

    updated = get_api_result(result_id, api_user["id"])
    return JSONResponse(content={
        "result_id": updated["id"],
        "alt_text": updated["alt_text"],
        "langbeschreibung": updated["langbeschreibung"],
        "bildtyp": updated["bildtyp"],
        "updated_at": updated["updated_at"],
    })


@app.get("/api/api-usage-stats")
async def api_usage_stats(user: dict = Depends(get_current_user)):
    """Get API usage statistics for the current user."""
    stats = get_api_usage_stats(user["id"])
    return stats


# ─── API Documentation ──────────────────────────────────────

@app.get("/api/v1/docs", response_class=HTMLResponse)
async def api_docs():
    """Accessible API documentation page (WCAG 2.2 AA)."""
    base = BASE_URL.rstrip("/")
    return """<!DOCTYPE html>
<html lang="de">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>InkluDocs API – Dokumentation</title>
<style>
:root { --primary: #1b2a4a; --accent: #e87722; --bg: #f8f9fa; --text: #1e293b; --muted: #64748b; --border: #e2e8f0; --code-bg: #1e293b; --code-text: #e2e8f0; }
*, *::before, *::after { box-sizing: border-box; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; color: var(--text); background: var(--bg); margin: 0; padding: 0; line-height: 1.6; }
a { color: var(--accent); }
a:focus-visible { outline: 3px solid var(--accent); outline-offset: 2px; border-radius: 2px; }
.container { max-width: 800px; margin: 0 auto; padding: 2rem 1.5rem; }
h1 { color: var(--primary); font-size: 1.8rem; margin-bottom: 0.5rem; }
h1 span { color: var(--accent); }
h2 { color: var(--primary); font-size: 1.3rem; margin-top: 2.5rem; padding-bottom: 0.3rem; border-bottom: 2px solid var(--accent); }
h3 { color: var(--primary); font-size: 1.1rem; margin-top: 1.5rem; }
.badge { display: inline-block; padding: 0.2rem 0.6rem; border-radius: 4px; font-size: 0.85rem; font-weight: 600; }
.badge-post { background: #16a34a; color: white; }
.badge-get { background: #2563eb; color: white; }
pre { background: var(--code-bg); color: var(--code-text); padding: 1.2rem; border-radius: 8px; overflow-x: auto; font-size: 0.9rem; line-height: 1.5; }
code { font-family: 'SF Mono', Consolas, 'Liberation Mono', Menlo, monospace; }
p code, li code { background: #e2e8f0; color: var(--primary); padding: 0.15rem 0.4rem; border-radius: 3px; font-size: 0.9em; }
table { width: 100%; border-collapse: collapse; margin: 1rem 0; }
th, td { text-align: left; padding: 0.6rem 0.8rem; border-bottom: 1px solid var(--border); }
th { background: var(--primary); color: white; font-weight: 600; }
tr:nth-child(even) { background: rgba(0,0,0,0.02); }
.note { padding: 1rem; background: #fff7ed; border-left: 4px solid var(--accent); border-radius: 0 4px 4px 0; margin: 1rem 0; }
footer { margin-top: 3rem; padding-top: 1rem; border-top: 1px solid var(--border); color: var(--muted); font-size: 0.85rem; }
.sr-only { position: absolute; width: 1px; height: 1px; padding: 0; margin: -1px; overflow: hidden; clip: rect(0,0,0,0); white-space: nowrap; border: 0; }
@media (prefers-color-scheme: dark) {
    :root { --bg: #0f172a; --text: #e2e8f0; --muted: #94a3b8; --border: #334155; --code-bg: #1e293b; }
    p code, li code { background: #334155; color: #e2e8f0; }
    tr:nth-child(even) { background: rgba(255,255,255,0.03); }
    .note { background: #1c1917; }
}
</style>
</head>
<body>
<main class="container" role="main">
<h1><span>Inklu</span>Docs API</h1>
<p style="color:var(--muted);">Version 1.0 &ndash; Alt-Text-Generierung fuer Bilder</p>

<nav aria-label="Inhaltsverzeichnis">
<h2 id="nav">Inhalt</h2>
<ul>
<li><a href="#auth">Authentifizierung</a></li>
<li><a href="#endpoint">Endpoint</a></li>
<li><a href="#request">Request</a></li>
<li><a href="#response">Response</a></li>
<li><a href="#get-result">Ergebnis abrufen (GET)</a></li>
<li><a href="#patch-result">Ergebnis bearbeiten (PATCH)</a></li>
<li><a href="#errors">Fehler-Codes</a></li>
<li><a href="#ratelimit">Rate-Limits</a></li>
<li><a href="#examples">Beispiele</a></li>
</ul>
</nav>

<h2 id="auth">Authentifizierung</h2>
<p>Alle API-Anfragen erfordern einen gueltigen API-Schluessel im <code>X-API-Key</code> Header.</p>
<p>API-Schluessel kannst du in der <a href="/app">InkluDocs-App</a> unter <strong>Einstellungen</strong> erstellen.</p>
<pre><code>X-API-Key: idocs_deinSchluesselHier</code></pre>

<h2 id="endpoint">Endpoint</h2>
<p><span class="badge badge-post">POST</span> <code>/api/v1/alt-text</code></p>
<p>Generiert einen barrierefreien Alt-Text fuer ein hochgeladenes Bild.</p>

<h2 id="request">Request</h2>
<p>Der Endpoint akzeptiert zwei Formate:</p>

<h3>Option A: Datei-Upload (Multipart)</h3>
<p>Content-Type: <code>multipart/form-data</code></p>
<table>
<caption class="sr-only">Request-Parameter Multipart</caption>
<thead><tr><th scope="col">Parameter</th><th scope="col">Typ</th><th scope="col">Pflicht</th><th scope="col">Beschreibung</th></tr></thead>
<tbody>
<tr><td><code>file</code></td><td>Datei</td><td>Ja</td><td>Bilddatei (JPG, PNG, GIF, WebP, BMP, TIFF, HEIC). Max. 10 MB.</td></tr>
<tr><td><code>context</code></td><td>Text</td><td>Nein</td><td>Umgebungstext fuer bessere Beschreibung (z.B. Bildunterschrift, Seitentitel).</td></tr>
<tr><td><code>language</code></td><td>Text</td><td>Nein</td><td>Sprache des Alt-Texts. Standard: <code>de</code></td></tr>
<tr><td><code>image_type</code></td><td>Text</td><td>Nein</td><td>Hinweis auf Bildtyp: <code>foto</code>, <code>diagramm</code>, <code>logo</code>, <code>icon</code>, <code>karte</code>, <code>screenshot</code>, <code>infografik</code>, <code>tabelle</code></td></tr>
</tbody>
</table>

<h3>Option B: Base64 (JSON)</h3>
<p>Content-Type: <code>application/json</code></p>
<table>
<caption class="sr-only">Request-Parameter JSON</caption>
<thead><tr><th scope="col">Parameter</th><th scope="col">Typ</th><th scope="col">Pflicht</th><th scope="col">Beschreibung</th></tr></thead>
<tbody>
<tr><td><code>image_base64</code></td><td>String</td><td>Ja</td><td>Bild als Base64-String. Data-URI-Prefix optional (z.B. <code>data:image/png;base64,...</code>).</td></tr>
<tr><td><code>context</code></td><td>String</td><td>Nein</td><td>Umgebungstext fuer bessere Beschreibung.</td></tr>
<tr><td><code>language</code></td><td>String</td><td>Nein</td><td>Sprache des Alt-Texts. Standard: <code>de</code></td></tr>
<tr><td><code>image_type</code></td><td>String</td><td>Nein</td><td>Hinweis auf Bildtyp.</td></tr>
</tbody>
</table>

<h2 id="response">Response</h2>
<p>Content-Type: <code>application/json</code></p>
<pre><code>{
  "result_id": "abc123xyz",
  "alt_text": "Beschreibung des Bildes",
  "langbeschreibung": "Ausfuehrliche Beschreibung...",
  "bildtyp": "foto",
  "konfidenz": "hoch",
  "processing_time_ms": 1234
}</code></pre>

<table>
<caption class="sr-only">Response-Felder</caption>
<thead><tr><th scope="col">Feld</th><th scope="col">Beschreibung</th></tr></thead>
<tbody>
<tr><td><code>result_id</code></td><td>Eindeutige ID dieses Ergebnisses – fuer GET und PATCH</td></tr>
<tr><td><code>alt_text</code></td><td>Der generierte Alt-Text (kurz, fuer das alt-Attribut)</td></tr>
<tr><td><code>langbeschreibung</code></td><td>Ausfuehrliche Beschreibung (fuer aria-describedby oder Langtext)</td></tr>
<tr><td><code>bildtyp</code></td><td>Erkannter Bildtyp (foto, diagramm, logo, etc.)</td></tr>
<tr><td><code>konfidenz</code></td><td>Vertrauen in die Erkennung: hoch, mittel, niedrig</td></tr>
<tr><td><code>processing_time_ms</code></td><td>Verarbeitungszeit in Millisekunden</td></tr>
</tbody>
</table>

<h2 id="get-result">Ergebnis abrufen</h2>
<p><span class="badge badge-get">GET</span> <code>/api/v1/alt-text/{result_id}</code></p>
<p>Ruft ein gespeichertes Ergebnis anhand seiner <code>result_id</code> ab. Erfordert denselben <code>X-API-Key</code> wie beim Erstellen.</p>
<pre><code>curl %%BASE_URL%%/api/v1/alt-text/abc123xyz \\
  -H "X-API-Key: idocs_deinSchluessel"</code></pre>
<p>Response-Felder: <code>result_id</code>, <code>alt_text</code>, <code>langbeschreibung</code>, <code>bildtyp</code>, <code>konfidenz</code>, <code>created_at</code>, <code>updated_at</code></p>

<h2 id="patch-result">Ergebnis bearbeiten</h2>
<p><span class="badge" style="background:#7c3aed;color:white;">PATCH</span> <code>/api/v1/alt-text/{result_id}</code></p>
<p>Aendert den Alt-Text und/oder die Langbeschreibung eines gespeicherten Ergebnisses. Mindestens eines der Felder muss angegeben werden.</p>
<pre><code>curl -X PATCH %%BASE_URL%%/api/v1/alt-text/abc123xyz \\
  -H "X-API-Key: idocs_deinSchluessel" \\
  -H "Content-Type: application/json" \\
  -d '{"alt_text": "Mein korrigierter Alt-Text"}'</code></pre>
<table>
<caption class="sr-only">PATCH Request-Felder</caption>
<thead><tr><th scope="col">Feld</th><th scope="col">Typ</th><th scope="col">Beschreibung</th></tr></thead>
<tbody>
<tr><td><code>alt_text</code></td><td>String</td><td>Neuer Alt-Text (optional, aber mindestens eines der beiden Felder)</td></tr>
<tr><td><code>langbeschreibung</code></td><td>String</td><td>Neue Langbeschreibung (optional)</td></tr>
</tbody>
</table>

<h2 id="errors">Fehler-Codes</h2>
<table>
<caption class="sr-only">HTTP-Fehler-Codes</caption>
<thead><tr><th scope="col">Code</th><th scope="col">Bedeutung</th></tr></thead>
<tbody>
<tr><td><code>400</code></td><td>Kein Bild mitgeschickt, ungueltiges Format oder fehlendes JSON-Feld</td></tr>
<tr><td><code>401</code></td><td>Ungueltiger oder fehlender API-Key</td></tr>
<tr><td><code>404</code></td><td>Ergebnis nicht gefunden (falsche result_id oder falscher API-Key)</td></tr>
<tr><td><code>413</code></td><td>Bild zu gross (max. 10 MB)</td></tr>
<tr><td><code>429</code></td><td>Rate-Limit ueberschritten</td></tr>
<tr><td><code>500</code></td><td>Interner Serverfehler</td></tr>
</tbody>
</table>

<h2 id="ratelimit">Rate-Limits</h2>
<p>Pro API-Schluessel gelten folgende Limits:</p>
<ul>
<li><strong>100 Bilder pro Tag</strong> (Alt-Text-Generierung, Reset um Mitternacht UTC)</li>
<li><strong>60 Anfragen pro Minute</strong></li>
</ul>
<p>Die verbleibenden Anfragen werden in Response-Headern mitgeteilt:</p>
<pre><code>X-RateLimit-Remaining-Minute: 58
X-RateLimit-Remaining-Day: 997</code></pre>
<p>Bei Ueberschreitung erhaeltst du HTTP <code>429</code> mit einem <code>Retry-After</code> Header.</p>

<h2 id="examples">Beispiele</h2>

<h3>Einfacher Aufruf mit curl</h3>
<pre><code>curl -X POST %%BASE_URL%%/api/v1/alt-text \\
  -H "X-API-Key: idocs_deinSchluessel" \\
  -F "file=@foto.jpg"</code></pre>

<h3>Mit Kontext fuer bessere Ergebnisse</h3>
<pre><code>curl -X POST %%BASE_URL%%/api/v1/alt-text \\
  -H "X-API-Key: idocs_deinSchluessel" \\
  -F "file=@diagramm.png" \\
  -F "context=Jahresbericht 2025, Kapitel Umsatzentwicklung" \\
  -F "image_type=diagramm"</code></pre>

<h3>Python-Beispiel</h3>
<pre><code>import requests

response = requests.post(
    "%%BASE_URL%%/api/v1/alt-text",
    headers={"X-API-Key": "idocs_deinSchluessel"},
    files={"file": open("bild.jpg", "rb")},
    data={"context": "Produktseite eines Online-Shops"},
)

data = response.json()
print(data["alt_text"])</code></pre>

<h3>JavaScript/Node.js-Beispiel</h3>
<pre><code>const form = new FormData();
form.append('file', fs.createReadStream('bild.jpg'));
form.append('context', 'Blog-Artikel ueber Barrierefreiheit');

const res = await fetch('%%BASE_URL%%/api/v1/alt-text', {
    method: 'POST',
    headers: { 'X-API-Key': 'idocs_deinSchluessel' },
    body: form,
});

const data = await res.json();
console.log(data.alt_text);</code></pre>

<h3>Base64-Beispiel (JSON)</h3>
<pre><code>curl -X POST %%BASE_URL%%/api/v1/alt-text \\
  -H "X-API-Key: idocs_deinSchluessel" \\
  -H "Content-Type: application/json" \\
  -d '{"image_base64": "data:image/jpeg;base64,/9j/4AAQ...", "context": "Startseite"}'</code></pre>

<div class="note" role="note">
<p><strong>Hinweis:</strong> Die API generiert Alt-Texte mit der gleichen KI-Pipeline wie die Web-App. Fuer beste Ergebnisse sende moeglichst viel Kontext im <code>context</code>-Feld mit (z.B. Seitentitel, umgebender Text, Bildunterschrift).</p>
</div>

<footer>
<p>InkluDocs API v1.0 &ndash; <a href="mailto:kontakt@inklutec.de">kontakt@inklutec.de</a> &ndash; <a href="/">Zurueck zu InkluDocs</a></p>
</footer>
</main>
</body>
</html>""".replace("%%BASE_URL%%", base)


# ─── InkluAgent (Chatbot pro Projekt) ────────────────────────

INKLUAGENT_ENABLED = os.environ.get("INKLUAGENT_ENABLED", "false").lower() in ("true", "1", "yes")


def _require_inkluagent():
    if not INKLUAGENT_ENABLED:
        raise HTTPException(status_code=404, detail="Nicht aktiviert")


def _require_project_owned(project_id: int, user_id: int):
    conn = get_db()
    try:
        row = conn.execute(
            "SELECT id FROM projects WHERE id = ? AND user_id = ?",
            (project_id, user_id),
        ).fetchone()
    finally:
        conn.close()
    if not row:
        raise HTTPException(status_code=404, detail="Projekt nicht gefunden")


@app.get("/api/projects/{project_id}/chat/history")
async def chat_get_history(project_id: int, user: dict = Depends(get_current_user)):
    _require_inkluagent()
    _require_project_owned(project_id, user["id"])
    from inkluagent import storage
    return {"messages": storage.get_history(project_id)}


@app.post("/api/projects/{project_id}/chat")
async def chat_send_message(project_id: int, request: Request, user: dict = Depends(get_current_user)):
    _require_inkluagent()
    _require_project_owned(project_id, user["id"])
    data = await request.json()
    message = (data.get("message") or "").strip()
    if not message:
        raise HTTPException(status_code=400, detail="Nachricht darf nicht leer sein")
    if len(message) > 5000:
        raise HTTPException(status_code=400, detail="Nachricht zu lang (max. 5000 Zeichen)")

    from inkluagent import storage
    from inkluagent.chat_engine import process_message

    storage.append_message(project_id, "user", message)
    result = await asyncio.get_event_loop().run_in_executor(
        None, process_message, project_id, message, user["id"]
    )
    storage.append_message(
        project_id, "assistant", result["reply"],
        image_refs=result.get("image_refs"),
        intent=result.get("intent"),
    )
    return {
        "reply": result["reply"],
        "intent": result.get("intent"),
        "image_refs": result.get("image_refs"),
        "actions": result.get("actions", []),
    }


@app.delete("/api/projects/{project_id}/chat")
async def chat_clear_history(project_id: int, user: dict = Depends(get_current_user)):
    _require_inkluagent()
    _require_project_owned(project_id, user["id"])
    from inkluagent import storage
    deleted = storage.clear_history(project_id)
    return {"deleted": deleted}


# ─── News / Neuigkeiten ─────────────────────────────────────


# === LINK-REFERENCE-POSTPROCESSOR (13.05.2026) ============================
# WCAG-relevant: Verlinkte Bilder bekommen "(verweist auf: ...)" an den
# Alt-Text gehaengt, deterministisch nach der LLM-Pipeline. Quelle ist die
# vom Web-Scraper extrahierte [Link-Beschriftung] im context_text.

_LINK_REF_GENERIC = {
    "mehr info", "mehr informationen", "weiterlesen", "hier klicken",
    "klicken", "link", "details", "mehr", "weiter", "mehr erfahren",
    "lesen sie mehr", "mehr lesen", "weiterlesen...",
}

def _append_link_reference(alt_text: str, context_text: str) -> str:
    """Haengt "(verweist auf: <Beschriftung>)" an alt_text, wenn das Bild
    verlinkt ist und eine sinnvolle Beschriftung im Kontext steht.

    Quellen im context_text (erste Treffer gewinnt):
    1. [Link-Beschriftung] <text>
    2. [title] verweist auf: <text>  (Scraper-vorformatiert)

    Skip-Regeln: leerer alt_text, kein Kontext, alt_text enthaelt bereits
    "verweist auf", Beschriftung generisch (Mehr Info etc.) oder <3 Zeichen.
    """
    if not alt_text or not context_text:
        return alt_text
    if "verweist auf" in alt_text.lower():
        return alt_text
    m = re.search(r"\[Link-Beschriftung\]\s*(.+?)(?:\n|$)", context_text)
    if m:
        label = m.group(1).strip()
        if label and len(label) >= 3 and label.lower() not in _LINK_REF_GENERIC:
            return f"{alt_text.rstrip()} (verweist auf: {label})"
    m = re.search(r"\[title\]\s*verweist auf:\s*(.+?)(?:\n|$)", context_text)
    if m:
        ref = m.group(1).strip()
        if ref:
            return f"{alt_text.rstrip()} (verweist auf: {ref})"
    return alt_text


NEUIGKEITEN = [
    # Neueste zuerst. Nur Eintraege, die fuer Nutzer relevant + auf Production live sind.
    {"datum": "25.06.2026", "text": "Passwort anzeigen: Bei Anmeldung, Registrierung und Passwort-Änderung kannst du dein Passwort jetzt per Häkchen sichtbar machen."},
    {"datum": "22.06.2026", "text": "Gastzugang: Du kannst ein Projekt mit einer Person ohne Konto teilen – sie kann deine Alt-Texte ansehen, kommentieren und freigeben."},
    {"datum": "19.06.2026", "text": "Seitentext übersichtlicher dargestellt; Alt-Texte und Beschreibungen lassen sich jetzt anhören."},
    {"datum": "10.06.2026", "text": "Mehrere PDFs in einem Projekt sammeln, einzeln umbenennen und löschen."},
    {"datum": "04.06.2026", "text": "Neu: Alt-Texte für ganze Webseiten. Adresse eingeben, die Bilder der Seite werden geladen – auch mehrere Seiten nacheinander."},
    {"datum": "02.06.2026", "text": "Neues Dashboard mit fester Seitenleiste: eine übersichtliche Startseite, von der aus du alles erreichst."},
    {"datum": "02.06.2026", "text": "Getrennte Werkzeuge für PDFs, Webseiten und Grafiken – du legst ein Projekt an und wählst das passende Werkzeug."},
    {"datum": "02.06.2026", "text": "PDF-Bearbeitung verbessert, mit Seitenvorschau für mehr Überblick."},
    {"datum": "02.06.2026", "text": "Verbesserte Barrierefreiheit: klarere Struktur, bessere Bedienung mit Screenreader und Tastatur."},
    {"datum": "13.05.2026", "text": "Chat-Assistent neu: Jedes Projekt hat jetzt einen eigenen Chatbot, der Bilder einsieht, Alt-Texte bewertet, Vorschläge macht und nach Bestätigung direkt in den Text übernimmt. Aktuell auf Deutsch."},
    {"datum": "13.05.2026", "text": "Mehrsprachigkeit in Vorbereitung: Englisch, Französisch und Spanisch werden in den kommenden Wochen nach und nach ausgerollt."},
    {"datum": "14.04.2026", "text": "Neue dreistufige Prüfpipeline aktiv: Klassifikation, Generierung und automatische Qualitätsprüfung gegen Halluzinationen. Laufende Auswertung zur weiteren Verbesserung."},
    {"datum": "07.04.2026", "text": "Alt-Text-Qualität verbessert: Produktbilder, Diagramme und verlinkte Bilder werden besser erkannt"},
    {"datum": "07.04.2026", "text": "Neue Bildformate: AVIF und HEIC werden jetzt unterstützt"},
    {"datum": "06.04.2026", "text": "Tageslimit: 100 Bilder pro Tag – Anzeige im Dashboard"},
    {"datum": "06.04.2026", "text": "Registrierung offen: Konto erstellen mit E-Mail-Bestätigung"},
    {"datum": "06.04.2026", "text": "InkluDocs unterstützen: Freiwillige Beiträge per PayPal möglich"},
]

def _news_date_key(d):
    """\"DD.MM.YYYY\" -> sortierbarer Schluessel \"YYYYMMDD\"."""
    import re as _re
    mm = _re.match(r"^(\d{2})\.(\d{2})\.(\d{4})$", d or "")
    return (mm.group(3) + mm.group(2) + mm.group(1)) if mm else (d or "")

def _newest_news_key():
    keys = [_news_date_key(n["datum"]) for n in NEUIGKEITEN]
    return max(keys) if keys else ""

@app.get("/api/news")
async def get_news(user: dict = Depends(get_current_user)):
    """Changelog-Eintraege + kontobezogener Lese-Stand (news_seen_until, Schluessel YYYYMMDD)."""
    conn = get_db()
    row = conn.execute("SELECT news_seen_until FROM users WHERE id = ?", (user["id"],)).fetchone()
    conn.close()
    seen = row["news_seen_until"] if row and row["news_seen_until"] else ""
    return {"news": NEUIGKEITEN, "seen_until": seen}

@app.post("/api/news/seen")
async def mark_news_seen(user: dict = Depends(get_current_user)):
    """Markiert alle aktuellen Neuigkeiten als gelesen (news_seen_until = neuester Schluessel)."""
    key = _newest_news_key()
    conn = get_db()
    conn.execute("UPDATE users SET news_seen_until = ? WHERE id = ?", (key, user["id"]))
    conn.commit()
    conn.close()
    return {"ok": True, "seen_until": key}

# ─── Frontend Routes ─────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    # Demo-Instanz: die Wurzel IST die oeffentliche Demo-Seite (statt Login/Dashboard).
    if demo_mod.demo_enabled():
        return FileResponse("/app/frontend/demo.html")
    lang = detect_language(request)
    registration_enabled = os.getenv("REGISTRATION_ENABLED", "true").lower() not in ("false", "0", "no")
    is_staging = "staging" in BASE_URL
    return templates.TemplateResponse(
        "index.html",
        template_context(
            request, lang,
            registration_enabled=registration_enabled,
            is_staging=is_staging,
        ),
    )


# 15.06.2026: Rechtsseiten der Demo IM Demo-Rahmen (Sidebar + Footer wie das
# Werkzeug). Bewusst eigene, NICHT login-geschuetzte Routen — die -app-Varianten
# des Werkzeugs setzen ein Login voraus und leiten anonyme Besucher zur Wurzel um.
# Der Rechtstext selbst kommt per fetch aus den oeffentlichen Routen (Single Source).
@app.get("/demo-impressum", response_class=HTMLResponse)
async def demo_impressum_page():
    if demo_mod.demo_enabled():
        return FileResponse("/app/frontend/demo-impressum.html")
    return RedirectResponse("/")


@app.get("/demo-datenschutz", response_class=HTMLResponse)
async def demo_datenschutz_page():
    if demo_mod.demo_enabled():
        return FileResponse("/app/frontend/demo-datenschutz.html")
    return RedirectResponse("/")


@app.get("/demo-nutzungsbedingungen", response_class=HTMLResponse)
async def demo_nutzungsbedingungen_page():
    if demo_mod.demo_enabled():
        return FileResponse("/app/frontend/demo-nutzungsbedingungen.html")
    return RedirectResponse("/")


@app.get("/set-language/{lang}")
async def set_language(lang: str, request: Request):
    """Wechsle UI-Sprache via Session-Cookie (spaeter auch in DB fuer User)."""
    referer = request.headers.get("referer", "/")
    # Zielort immer zurueck zum Referer, aber nur wenn same-origin (Sicherheit)
    if referer and (referer.startswith(BASE_URL) or referer.startswith("/")):
        redirect_to = referer
    else:
        redirect_to = "/"
    response = RedirectResponse(redirect_to, status_code=303)
    if lang in SUPPORTED_LANGUAGES:
        # 1 Jahr gueltig, samesite=lax, nicht HttpOnly (Client koennte lesen)
        response.set_cookie("lang", lang, max_age=365*24*60*60, samesite="lax")
    return response

@app.get("/register", response_class=HTMLResponse)
async def register_page():
    if os.getenv("REGISTRATION_ENABLED", "true").lower() in ("false", "0", "no"):
        return """<!DOCTYPE html><html lang="de"><head><meta charset="utf-8"><title>InkluDocs - Registrierung geschlossen</title>
        <link rel="stylesheet" href="/static/style.css"></head><body>
        <div class="auth-wrapper"><div class="auth-container" role="main" aria-label="Registrierung geschlossen">
        <h1><span class="brand">Inklu</span>Docs</h1>
        <p style="margin:2rem 0;font-size:1.1rem;">Die Registrierung ist derzeit geschlossen.</p>
        <p>InkluDocs befindet sich in der geschlossenen Beta-Phase. Wenn du einen Testzugang erhalten moechtest, schreib bitte eine E-Mail an <strong>kontakt@inklutec.de</strong>.</p>
        <p style="margin-top:2rem;"><a href="/">Zurueck zur Anmeldung</a></p>
        </div></div></body></html>"""
    html = open("/app/frontend/register.html").read()
    if "staging" in BASE_URL:
        html = html.replace("<title>InkluDocs", "<title>InkluDocs (Testumgebung)")
        html = html.replace('<span class="brand">Inklu</span>Docs', '<span class="brand">Inklu</span>Docs <span style="font-size:0.6em;color:#e87722;font-weight:normal;">(Testumgebung)</span>')
    return html

@app.get("/forgot", response_class=HTMLResponse)
async def forgot_page():
    return open("/app/frontend/forgot.html").read()

@app.get("/reset", response_class=HTMLResponse)
async def reset_page():
    return open("/app/frontend/reset.html").read()

def _serve_protected_page(request: Request, filename: str):
    """Liefert eine login-geschuetzte HTML-Seite aus dem frontend-Verzeichnis.
    Leitet zu / um, wenn kein gueltiges Login-Cookie vorliegt."""
    token = request.cookies.get("token")
    if not token:
        return RedirectResponse("/")
    try:
        jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except JWTError:
        return RedirectResponse("/")
    html = open(f"/app/frontend/{filename}").read()
    if "staging" in BASE_URL:
        html = html.replace("<title>InkluDocs</title>", "<title>InkluDocs (Testumgebung)</title>")
        html = html.replace('<span class="brand">Inklu</span>Docs', '<span class="brand">Inklu</span>Docs <span style="font-size:0.6em;color:#c75000;font-weight:normal;">(Testumgebung)</span>')
    return html


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard_page(request: Request):
    return _serve_protected_page(request, "dashboard.html")


@app.get("/projekte", response_class=HTMLResponse)
async def projects_page(request: Request):
    return _serve_protected_page(request, "projekte.html")


@app.get("/projekt-neu", response_class=HTMLResponse)
async def new_project_page(request: Request):
    return _serve_protected_page(request, "projekt-neu.html")


@app.get("/einstellungen", response_class=HTMLResponse)
async def settings_page(request: Request):
    return _serve_protected_page(request, "einstellungen.html")


@app.get("/benutzer", response_class=HTMLResponse)
async def users_page(request: Request):
    return _serve_protected_page(request, "benutzer.html")


@app.get("/datensicherheit", response_class=HTMLResponse)
async def datensicherheit_page(request: Request):
    # Steve/Karbe 08.06.2026: eigene In-App-Sicht der Datenschutzerklaerung,
    # damit die Sidebar beim Aufruf sichtbar bleibt. Der Inhalt wird im
    # Frontend per fetch aus /datenschutz (Single Source) geladen.
    return _serve_protected_page(request, "datensicherheit.html")


@app.get("/impressum-app", response_class=HTMLResponse)
async def impressum_app_page(request: Request):
    # Steve 08.06.2026: In-App-Sicht des Impressums (Schwester zu /datensicherheit).
    # Wird vom Footer der eingeloggten App-Bereiche statt /impressum verlinkt, damit
    # Dashboard/Sidebar beim Klick sichtbar bleiben. Inhalt kommt per fetch aus
    # /impressum (Single Source — keine Doppelpflege).
    return _serve_protected_page(request, "impressum-app.html")


@app.get("/nutzungsbedingungen-app", response_class=HTMLResponse)
async def nutzungsbedingungen_app_page(request: Request):
    # Steve 08.06.2026: In-App-Sicht der Nutzungsbedingungen. Gleiches Muster
    # wie /datensicherheit und /impressum-app. Inhalt aus /nutzungsbedingungen
    # per fetch geladen.
    return _serve_protected_page(request, "nutzungsbedingungen-app.html")


@app.get("/app", response_class=HTMLResponse)
async def app_page(request: Request):
    token = request.cookies.get("token")
    if not token:
        return RedirectResponse("/")
    try:
        jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except JWTError:
        return RedirectResponse("/")
    html = open("/app/frontend/app.html").read()
    if "staging" in BASE_URL:
        html = html.replace("<title>InkluDocs</title>", "<title>InkluDocs (Testumgebung)</title>")
        html = html.replace('<span class="brand">Inklu</span>Docs', '<span class="brand">Inklu</span>Docs <span style="font-size:0.6em;color:#e87722;font-weight:normal;">(Testumgebung)</span>')
    return html


# ============================================================
#  Gastzugang / Projekt-Freigabe — Routen (19.06.2026, Etappe 2a)
#  Alle additiv. Kein bestehender (Login-)Pfad wird veraendert.
# ============================================================
@app.get("/freigabe/{token}", response_class=HTMLResponse)
async def freigabe_page(token: str, request: Request):
    """Gast-Eingang: liefert dieselbe app.html im GAST-MODUS (kein Login noetig).
    Ungueltiger/abgelaufener/widerrufener Token -> neutraler Hinweis (kein Datenleck).
    Der Datenzugriff ist zusaetzlich durch das E-Mail-Gate geschuetzt."""
    if not sharing.is_valid_share(token):
        return HTMLResponse(
            "<!doctype html><html lang='de'><head><meta charset='utf-8'>"
            "<title>InkluDocs</title></head>"
            "<body style='font-family:sans-serif;max-width:40rem;margin:3rem auto;padding:0 1rem'>"
            "<h1>Link ungueltig oder abgelaufen</h1>"
            "<p>Diese Freigabe ist nicht (mehr) gueltig. Bitte wenden Sie sich an die Person, "
            "die Ihnen den Link geschickt hat.</p></body></html>",
            status_code=404,
        )
    html = open("/app/frontend/app.html").read()
    # GAST-MODUS-Flag + Token vor den Skripten einspeisen (Token ist URL-safe).
    inject = '<script>window.GUEST_MODE=true;window.SHARE_TOKEN="' + token + '";</script>'
    html = html.replace("</head>", inject + "</head>", 1)
    if "staging" in BASE_URL:
        html = html.replace("<title>InkluDocs</title>", "<title>InkluDocs (Testumgebung)</title>")
        html = html.replace('<span class="brand">Inklu</span>Docs', '<span class="brand">Inklu</span>Docs <span style="font-size:0.6em;color:#e87722;font-weight:normal;">(Testumgebung)</span>')
    return html


@app.post("/api/freigabe/{token}/confirm")
async def freigabe_confirm(token: str, request: Request):
    """E-Mail-Gate: prueft die eingegebene E-Mail gegen die Einladung. Bei Erfolg
    wird ein guest_token-Cookie gesetzt (an genau diese Freigabe gebunden)."""
    data = await request.json()
    email = (data.get("email") or "").strip()
    if not sharing.confirm_guest_email(token, email):
        raise HTTPException(status_code=403, detail="E-Mail stimmt nicht mit der Einladung ueberein.")
    resp = JSONResponse({"ok": True})
    resp.set_cookie("guest_token", create_guest_token(token, email.lower()),
                    httponly=True, samesite="strict", max_age=TOKEN_EXPIRE_HOURS * 3600)
    return resp


def _require_guest(request, token):
    """Gemeinsamer Wachposten: gueltige Gast-Sitzung + nutzbare Freigabe -> share,
    sonst HTTPException(401)."""
    session = get_guest_session(request, token)
    share = sharing.get_share(token) if session else None
    if not session or not share or share["status"] not in ("active", "completed"):
        raise HTTPException(status_code=401, detail="Bitte zuerst die E-Mail bestaetigen.")
    return share


@app.get("/api/freigabe/{token}")
async def freigabe_data(token: str, request: Request):
    """Projektdaten fuer den Gast — nur mit gueltiger Gast-Sitzung, nur das eine
    Projekt dieser Freigabe. Spiegelt get_project, aber token- statt user-begrenzt."""
    share = _require_guest(request, token)
    pid = share["project_id"]
    conn = get_db()
    project = conn.execute("SELECT * FROM projects WHERE id = ?", (pid,)).fetchone()
    if not project:
        conn.close()
        raise HTTPException(status_code=404, detail="Projekt nicht gefunden.")
    images = conn.execute(
        """SELECT i.*, (SELECT body FROM messages m WHERE m.image_id = i.id AND m.msg_type = 'review_note' LIMIT 1) AS review_note FROM images i
           LEFT JOIN documents d ON d.id = i.document_id
           WHERE i.project_id = ?
           ORDER BY COALESCE(d.doc_index, 0), i.page_number, i.image_index""",
        (pid,)
    ).fetchall()
    documents = conn.execute(
        """SELECT id, doc_index, original_filename, display_name, extraction_method, total_images, created_at
           FROM documents WHERE project_id = ? ORDER BY doc_index""",
        (pid,)
    ).fetchall()
    conn.close()
    proj_dict = dict(project)
    _is_pdf = (proj_dict.get("tool") == "pdf" or proj_dict.get("project_type") == "pdf")
    return {
        "project": proj_dict,
        "images": [dict(i) for i in images],
        "documents": [dict(d) for d in documents],
        "show_langbeschreibung": (not _is_pdf) or pdf_langbeschreibung_enabled(),
        "guest": True,
    }


@app.get("/api/freigabe/{token}/images/{image_id}/file")
async def freigabe_image_file(token: str, image_id: int, request: Request):
    """Bild-Datei fuer den Gast — gast-begrenzt + nur Bilder DES freigegebenen Projekts."""
    share = _require_guest(request, token)
    conn = get_db()
    img = conn.execute("SELECT * FROM images WHERE id = ? AND project_id = ?",
                       (image_id, share["project_id"])).fetchone()
    conn.close()
    if not img or not os.path.exists(img["image_path"]):
        raise HTTPException(status_code=404, detail="Bild nicht gefunden")
    return FileResponse(img["image_path"])


@app.get("/api/freigabe/{token}/images/{image_id}/page-view")
async def freigabe_image_page_view(token: str, image_id: int, request: Request):
    """Seitenansicht-PNG fuer den Gast — gast-begrenzt + projektgebunden."""
    share = _require_guest(request, token)
    conn = get_db()
    img = conn.execute("SELECT page_view_path FROM images WHERE id = ? AND project_id = ?",
                       (image_id, share["project_id"])).fetchone()
    conn.close()
    if not img or not img["page_view_path"] or not os.path.exists(img["page_view_path"]):
        raise HTTPException(status_code=404, detail="Seitenansicht nicht gefunden")
    return FileResponse(img["page_view_path"])


@app.post("/api/freigabe/{token}/images/{image_id}/alttext")
async def freigabe_save_alttext(token: str, image_id: int, request: Request):
    """Gast bearbeitet den Alt-Text haendisch (lesen + editieren erlaubt, KEINE KI).
    Speichert in alt_text_edited — nur fuer Bilder DES freigegebenen Projekts.
    Spiegelt update_alt_text (Besitzer), aber token-/gast-begrenzt."""
    share = _require_guest(request, token)
    data = await request.json()
    conn = get_db()
    img = conn.execute("SELECT id FROM images WHERE id = ? AND project_id = ?",
                       (image_id, share["project_id"])).fetchone()
    if not img:
        conn.close()
        raise HTTPException(status_code=404, detail="Bild nicht gefunden")
    conn.execute("UPDATE images SET alt_text_edited = ? WHERE id = ?",
                 (data.get("alt_text", ""), image_id))
    if "langbeschreibung" in data:
        conn.execute("UPDATE images SET langbeschreibung = ? WHERE id = ?",
                     (data.get("langbeschreibung", ""), image_id))
    conn.commit()
    conn.close()
    return {"ok": True}


@app.post("/api/freigabe/{token}/images/{image_id}/review")
async def freigabe_set_review(token: str, image_id: int, request: Request):
    """Gast setzt den Pruefstatus eines Bildes (offen/freigegeben/zu_ueberarbeiten)
    und optional einen Kommentar. Genau EINE review_note pro Bild (in messages);
    leerer Kommentar loescht sie. Alles gast-/projektbegrenzt."""
    share = _require_guest(request, token)
    session = get_guest_session(request, token)
    data = await request.json()
    status = data.get("status", "offen")
    if status not in ("offen", "freigegeben", "zu_ueberarbeiten"):
        raise HTTPException(status_code=400, detail="Ungueltiger Status.")
    comment = (data.get("comment") or "").strip()
    conn = get_db()
    img = conn.execute("SELECT id FROM images WHERE id = ? AND project_id = ?",
                       (image_id, share["project_id"])).fetchone()
    if not img:
        conn.close()
        raise HTTPException(status_code=404, detail="Bild nicht gefunden")
    conn.execute("UPDATE images SET review_status = ?, reviewed_at = datetime('now') WHERE id = ?",
                 (status, image_id))
    # Genau eine review_note pro Bild: alte ersetzen / bei leerem Kommentar entfernen.
    conn.execute("DELETE FROM messages WHERE image_id = ? AND msg_type = 'review_note'", (image_id,))
    if comment:
        conn.execute(
            "INSERT INTO messages (project_id, image_id, sender_guest, recipient_user_id, body, msg_type) "
            "VALUES (?, ?, ?, ?, ?, 'review_note')",
            (share["project_id"], image_id, (session or {}).get("guest", ""), share["created_by"], comment),
        )
    conn.commit()
    conn.close()
    return {"ok": True, "review_status": status, "comment": comment}


def _mail_escape(t):
    """Minimaler HTML-Schutz fuer Gast-Eingaben in Benachrichtigungs-Mails."""
    return (t or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


# ─── Etappe 4: Besitzer erstellt/verwaltet Freigaben ───
@app.post("/api/projects/{project_id}/share")
async def create_project_share(project_id: int, request: Request, user: dict = Depends(get_current_user)):
    """Besitzer legt eine Freigabe (Gastzugang) an. Bei notify=true (Standard) verschickt der
    Server die Einladung serverseitig: From-Anzeigename "<Besitzer> ueber InkluDocs", Reply-To
    = Besitzer (Antworten gehen an ihn, NICHT an InkluTec). Bei notify=false wird NICHT versendet
    (Besitzer holt sich nur den Link und mailt selbst). Optionale persoenliche Nachricht ersetzt
    nur den Einleitungssatz; Link + Bestaetigungs-Hinweis bleiben fest."""
    import html
    data = await request.json()
    guest_email = (data.get("guest_email") or "").strip()
    guest_name = (data.get("guest_name") or "").strip()
    custom_message = (data.get("message") or "").strip()
    notify = data.get("notify", True)
    if not guest_email or "@" not in guest_email:
        raise HTTPException(status_code=400, detail="Bitte eine gueltige E-Mail-Adresse angeben.")
    conn = get_db()
    proj = conn.execute("SELECT id, name, filename FROM projects WHERE id = ? AND user_id = ?",
                        (project_id, user["id"])).fetchone()
    conn.close()
    if not proj:
        raise HTTPException(status_code=404, detail="Projekt nicht gefunden")
    token = sharing.create_share(project_id, guest_email, user["id"], guest_name=guest_name)
    url = BASE_URL + "/freigabe/" + token
    sent = False
    if notify:
        owner = get_user_by_id(user["id"]) or {}
        owner_name = (owner.get("display_name") or "").strip() or "Ein InkluDocs-Nutzer"
        owner_email = owner.get("email") or user["email"]
        proj_name = (proj["name"] or proj["filename"] or ("Projekt " + str(project_id)))
        greet = ("Hallo " + guest_name + ",") if guest_name else "Hallo,"
        if custom_message:
            intro = html.escape(custom_message).replace("\n", "<br>")
        else:
            intro = (html.escape(owner_name) + " möchte Sie bitten, die Alt-Texte für das Projekt „"
                     + html.escape(proj_name) + "“ zu prüfen.")
        body = (
            "<p>" + greet + "</p>"
            "<p>" + intro + "</p>"
            "<p>Bitte öffnen Sie diesen Link und bestätigen Sie Ihre E-Mail-Adresse:</p>"
            "<p><a href='" + url + "'>" + url + "</a></p>"
            "<p>Eingeladen von: " + html.escape(owner_email) + "</p>"
            "<p>Vielen Dank!</p>"
        )
        subject = "Bitte Alt-Texte prüfen: " + proj_name
        sent = send_email(guest_email, subject, body, bcc_admin=False,
                          reply_to=owner_email, from_name=(owner_name + " über InkluDocs"))
    return {"ok": True, "token": token, "url": url, "guest_email": guest_email, "sent": bool(sent)}


@app.get("/api/projects/{project_id}/shares")
async def list_project_shares(project_id: int, user: dict = Depends(get_current_user)):
    conn = get_db()
    proj = conn.execute("SELECT id FROM projects WHERE id = ? AND user_id = ?",
                        (project_id, user["id"])).fetchone()
    conn.close()
    if not proj:
        raise HTTPException(status_code=404, detail="Projekt nicht gefunden")
    shares = sharing.list_shares_for_project(project_id, user["id"])
    for sh in shares:
        sh["url"] = BASE_URL + "/freigabe/" + sh["token"]
    return {"shares": shares}


@app.post("/api/projects/{project_id}/shares/revoke")
async def revoke_project_share(project_id: int, request: Request, user: dict = Depends(get_current_user)):
    data = await request.json()
    if not sharing.revoke_share(data.get("token", ""), user["id"]):
        raise HTTPException(status_code=404, detail="Freigabe nicht gefunden")
    return {"ok": True}


# ─── Etappe 5: Gast schliesst die Pruefung ab -> Besitzer wird benachrichtigt ───
@app.post("/api/freigabe/{token}/complete")
async def freigabe_complete(token: str, request: Request):
    """Gast schliesst die Pruefung ab: Freigabe -> 'completed', Zusammenfassung +
    optionale Nachricht per E-Mail an den Besitzer. Bleibt danach weiter lesbar
    (Gast kann jederzeit zurueckkommen)."""
    share = _require_guest(request, token)
    session = get_guest_session(request, token)
    data = await request.json()
    message = (data.get("message") or "").strip()
    pid = share["project_id"]
    conn = get_db()
    project = conn.execute("SELECT * FROM projects WHERE id = ?", (pid,)).fetchone()
    imgs = conn.execute("SELECT review_status FROM images WHERE project_id = ?", (pid,)).fetchall()
    notes = conn.execute(
        "SELECT body FROM messages WHERE project_id = ? AND msg_type = 'review_note'", (pid,)
    ).fetchall()
    conn.close()
    freigegeben = sum(1 for i in imgs if i["review_status"] == "freigegeben")
    zu_ueber = sum(1 for i in imgs if i["review_status"] == "zu_ueberarbeiten")
    offen = sum(1 for i in imgs if (i["review_status"] or "offen") == "offen")
    sharing.mark_completed(token)
    from database import get_user_by_id
    owner = get_user_by_id(share["created_by"])
    pname = (project["name"] if project and project["name"] else (project["filename"] if project else "Projekt"))
    guest = (session or {}).get("guest", share["guest_email"])
    link = BASE_URL + "/freigabe/" + token
    notes_html = "".join("<li>" + _mail_escape(n["body"]) + "</li>" for n in notes)
    body = (
        "<p>" + _mail_escape(guest) + " hat die Pruefung fuer das Projekt &bdquo;"
        + _mail_escape(pname) + "&ldquo; abgeschlossen.</p>"
        + "<p><strong>Ergebnis:</strong> " + str(freigegeben) + " freigegeben, "
        + str(zu_ueber) + " zu ueberarbeiten, " + str(offen) + " offen.</p>"
        + (("<p><strong>Nachricht:</strong><br>" + _mail_escape(message).replace(chr(10), "<br>") + "</p>") if message else "")
        + (("<p><strong>Anmerkungen:</strong></p><ul>" + notes_html + "</ul>") if notes_html else "")
        + "<p>Im Werkzeug ansehen: <a href='" + link + "'>" + link + "</a></p>"
    )
    if owner and owner.get("email"):
        send_email(owner["email"], "InkluDocs: Pruefung abgeschlossen (" + pname + ")", body, bcc_admin=False)
    return {"ok": True, "freigegeben": freigegeben, "zu_ueberarbeiten": zu_ueber, "offen": offen}


@app.get("/api/review-overview")
async def review_overview(user: dict = Depends(get_current_user)):
    """Projekte des Besitzers, die zur Pruefung freigegeben sind, mit Review-Zaehlern
    fuers Dashboard-Panel (offen / zu ueberarbeiten / freigegeben)."""
    conn = get_db()
    rows = conn.execute(
        """SELECT p.id AS id, p.name AS name, p.filename AS filename,
                  SUM(CASE WHEN COALESCE(i.review_status,'offen')='offen' THEN 1 ELSE 0 END) AS offen,
                  SUM(CASE WHEN i.review_status='zu_ueberarbeiten' THEN 1 ELSE 0 END) AS zu_ueberarbeiten,
                  SUM(CASE WHEN i.review_status='freigegeben' THEN 1 ELSE 0 END) AS freigegeben,
                  COUNT(i.id) AS total,
                  (SELECT GROUP_CONCAT(DISTINCT COALESCE(NULLIF(g.guest_name,''), g.guest_email)) FROM shares g
                    WHERE g.project_id = p.id AND g.status IN ('active','completed')) AS guests
           FROM projects p
           JOIN images i ON i.project_id = p.id
           WHERE p.user_id = ?
             AND EXISTS (SELECT 1 FROM shares s WHERE s.project_id = p.id AND s.status IN ('active','completed'))
           GROUP BY p.id
           ORDER BY p.updated_at DESC""",
        (user["id"],)
    ).fetchall()
    conn.close()
    projects = []
    for r in rows:
        d = dict(r)
        d["display_name"] = (d.get("name") or d.get("filename") or ("Projekt " + str(d["id"])))
        projects.append(d)
    return {"projects": projects}

@app.get("/impressum", response_class=HTMLResponse)
async def impressum_page():
    return open("/app/frontend/impressum.html").read()

@app.get("/datenschutz", response_class=HTMLResponse)
async def datenschutz_page():
    return open("/app/frontend/datenschutz.html").read()

@app.get("/nutzungsbedingungen", response_class=HTMLResponse)
async def nutzungsbedingungen_page():
    return open("/app/frontend/nutzungsbedingungen.html").read()
