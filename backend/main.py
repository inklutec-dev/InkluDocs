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
from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, Request, Form
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from jose import jwt, JWTError

from database import (
    init_db, get_db, create_user, verify_user, get_user_by_email, get_user_by_id,
    create_password_reset_token, verify_reset_token, reset_password,
    list_all_users, update_user_active, delete_user_data, admin_reset_password,
    create_api_key, verify_api_key, list_api_keys, delete_api_key, rename_api_key,
    log_api_usage, get_api_usage_stats,
    create_email_change_token, confirm_email_change,
    create_api_result, get_api_result, update_api_result,
    create_email_verification_token, mark_email_verified, resend_verification_token,
    get_daily_image_count, get_daily_api_count,
)
from pdf_processor import extract_images_from_pdf, generate_alt_text, generate_alt_text_for_image, clear_project_cache
from i18n import get_templates, detect_language, template_context, SUPPORTED_LANGUAGES

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


def send_email(to_email: str, subject: str, html_body: str, bcc_admin: bool = True, attachment_path: str = None) -> bool:
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
        msg["From"] = f"InkluDocs <{SMTP_FROM}>"
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
    user = get_current_user(request)
    if not user.get("is_admin"):
        raise HTTPException(status_code=403, detail="Nur fuer Administratoren")
    return user


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
        raise HTTPException(status_code=403, detail="E-Mail-Adresse noch nicht bestaetigt. Bitte pruefen Sie Ihr Postfach oder fordern Sie einen neuen Bestaetigungslink an.")

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
<p style="color:#64748b;font-size:0.9rem;">Die E-Mail-Adresse wurde noch nicht bestaetigt.</p>
</body></html>"""
    send_email(NOTIFICATION_EMAIL, "InkluDocs: Neue Registrierung", admin_body, bcc_admin=False)

    return JSONResponse({"ok": True, "message": "Bestaetigungslink wurde gesendet. Bitte pruefen Sie Ihr Postfach."})


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
async def admin_toggle_active(user_id: int, user: dict = Depends(require_admin)):
    target = get_user_by_id(user_id)
    if not target:
        raise HTTPException(status_code=404, detail="User nicht gefunden")
    if target["id"] == user["id"]:
        raise HTTPException(status_code=400, detail="Sie koennen sich nicht selbst deaktivieren")
    new_status = 0 if target["is_active"] else 1
    update_user_active(user_id, new_status)
    return {"ok": True, "is_active": new_status}


@app.post("/api/admin/users/{user_id}/reset-password")
async def admin_reset_user_password(user_id: int, request: Request, user: dict = Depends(require_admin)):
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
async def admin_create_user(request: Request, user: dict = Depends(require_admin)):
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
async def admin_delete_user(user_id: int, user: dict = Depends(require_admin)):
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


@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...), user: dict = Depends(get_current_user)):
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
        return await _handle_pdf_upload(file_path, filename, user)
    else:
        return await _handle_image_upload(file_path, filename, user, content, ext)


async def _handle_pdf_upload(file_path: str, filename: str, user: dict) -> dict:
    """Process a PDF upload (existing behavior)."""
    conn = get_db()
    cursor = conn.execute(
        "INSERT INTO projects (user_id, filename, original_path, status, project_type) VALUES (?, ?, ?, 'extracting', 'pdf')",
        (user["id"], filename, file_path)
    )
    project_id = cursor.lastrowid
    conn.commit()

    # Extract images
    img_dir = os.path.join(RESULTS_DIR, str(user["id"]), str(project_id))
    os.makedirs(img_dir, exist_ok=True)

    try:
        images = extract_images_from_pdf(file_path, img_dir, project_id)
    except Exception as e:
        conn.execute("UPDATE projects SET status = 'error' WHERE id = ?", (project_id,))
        conn.commit()
        conn.close()
        raise HTTPException(status_code=500, detail=f"PDF-Verarbeitung fehlgeschlagen: {str(e)}")

    # Store images in DB (including bounding box for vector graphics support)
    for img in images:
        bbox = img.get("bbox")
        bbox_x0 = bbox[0] if bbox else None
        bbox_y0 = bbox[1] if bbox else None
        bbox_x1 = bbox[2] if bbox else None
        bbox_y1 = bbox[3] if bbox else None
        is_vector = 1 if img.get("is_vector") else 0

        conn.execute(
            """INSERT INTO images (project_id, page_number, image_index, image_path, context_text,
               width, height, xref, bbox_x0, bbox_y0, bbox_x1, bbox_y1, is_vector,
               original_alt, page_view_path, page_text)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (project_id, img["page_number"], img["image_index"], img["image_path"],
             img["context_text"], img["width"], img["height"], img["xref"],
             bbox_x0, bbox_y0, bbox_x1, bbox_y1, is_vector,
             img.get("original_alt", ""),
             img.get("page_view_path", ""), img.get("page_text", ""))
        )

    # PDFIX-INTEGRATION (24.04.2026): Extraktionsweg merken (fitz|pdfix)
    extraction_method = "pdfix" if images and any(i.get("source") == "pdfix" for i in images) else "fitz"
    conn.execute(
        "UPDATE projects SET status = 'extracted', total_images = ?, extraction_method = ? WHERE id = ?",
        (len(images), extraction_method, project_id)
    )
    conn.commit()
    conn.close()

    return {
        "ok": True,
        "project_id": project_id,
        "filename": filename,
        "total_images": len(images),
        "project_type": "pdf",
        "extraction_method": extraction_method,
    }


async def _handle_image_upload(file_path: str, filename: str, user: dict, content: bytes, ext: str) -> dict:
    """Process a direct image upload."""
    from PIL import Image as PILImage
    try:
        from pillow_heif import register_heif_opener
        register_heif_opener()
    except ImportError:
        pass

    conn = get_db()
    cursor = conn.execute(
        "INSERT INTO projects (user_id, filename, original_path, status, project_type) VALUES (?, ?, ?, 'extracted', 'images')",
        (user["id"], filename, file_path)
    )
    project_id = cursor.lastrowid
    conn.commit()

    # Create project image directory
    img_dir = os.path.join(RESULTS_DIR, str(user["id"]), str(project_id))
    os.makedirs(img_dir, exist_ok=True)

    # Convert HEIC/HEIF/BMP/TIFF to JPEG for model compatibility
    needs_conversion = ext in {".heic", ".heif", ".bmp", ".tiff", ".tif"}
    if needs_conversion:
        img_path = os.path.join(img_dir, "img_1.jpg")
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
        img_path = os.path.join(img_dir, f"img_1{ext}")
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
           VALUES (?, 1, 1, ?, '', ?, ?, 0)""",
        (project_id, img_path, width, height)
    )
    conn.execute(
        "UPDATE projects SET total_images = 1 WHERE id = ?",
        (project_id,)
    )
    conn.commit()
    conn.close()

    return {
        "ok": True,
        "project_id": project_id,
        "filename": filename,
        "total_images": 1,
        "project_type": "images",
    }


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

    if not url:
        raise HTTPException(status_code=400, detail="Bitte eine URL eingeben")

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

    # Create project
    conn = get_db()
    page_title = soup.title.string.strip() if soup.title and soup.title.string else parsed.netloc
    cursor = conn.execute(
        "INSERT INTO projects (user_id, filename, original_path, status, project_type, source_url) VALUES (?, ?, '', 'extracting', 'url', ?)",
        (user["id"], f"Website: {page_title[:80]}", url)
    )
    project_id = cursor.lastrowid
    conn.commit()

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
        for idx, img_tag in enumerate(img_tags, 1):
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

    if downloaded == 0:
        conn.execute("UPDATE projects SET status = 'error' WHERE id = ?", (project_id,))
        conn.commit()
        conn.close()
        raise HTTPException(status_code=404, detail="Keine Bilder konnten heruntergeladen werden")

    conn.execute(
        "UPDATE projects SET status = 'extracted', total_images = ? WHERE id = ?",
        (downloaded, project_id)
    )
    conn.commit()
    conn.close()

    return {
        "ok": True,
        "project_id": project_id,
        "total_images": downloaded,
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


@app.get("/api/projects/{project_id}")
async def get_project(project_id: int, user: dict = Depends(get_current_user)):
    conn = get_db()
    project = conn.execute(
        "SELECT * FROM projects WHERE id = ? AND user_id = ?", (project_id, user["id"])
    ).fetchone()
    if not project:
        conn.close()
        raise HTTPException(status_code=404, detail="Projekt nicht gefunden")

    images = conn.execute(
        "SELECT * FROM images WHERE project_id = ? ORDER BY page_number, image_index", (project_id,)
    ).fetchall()
    conn.close()

    return {
        "project": dict(project),
        "images": [dict(img) for img in images],
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

    processed = 0
    for img in images:
        conn.execute("UPDATE images SET status = 'processing' WHERE id = ?", (img["id"],))
        conn.commit()

        try:
            # v2.2: Pass width, height, original_alt to pipeline
            img_width = img["width"] if img["width"] else 0
            img_height = img["height"] if img["height"] else 0
            img_original_alt = img["original_alt"] if img["original_alt"] else ""

            # First pass: general prompt for type detection + alt-text
            result = await asyncio.get_event_loop().run_in_executor(
                None, generate_alt_text, img["image_path"], img["context_text"], None,
                img_width, img_height, img_original_alt
            )

            # Second pass: if complex type detected, re-generate with specialized prompt
            # to get langbeschreibung automatically
            from context_engine import is_complex_type
            detected_type = result.get("bildtyp", "")
            langbeschreibung = result.get("langbeschreibung", "")

            if is_complex_type(detected_type) and not langbeschreibung:
                specialized_result = await asyncio.get_event_loop().run_in_executor(
                    None, generate_alt_text, img["image_path"], img["context_text"], detected_type,
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
                   needs_review = ?, pipeline_steps = ?, validation_result = ?, status = 'done' WHERE id = ?""",
                (_append_link_reference(result["alt_text"], img["context_text"] or ""), result["bildtyp"], result.get("konfidenz", "mittel"), langbeschreibung,
                 1 if result.get("needs_review") else 0,
                 result.get("pipeline_steps", ""),
                 result.get("validation_result", ""),
                 img["id"])
            )
        except Exception as e:
            import traceback
            print(f"Fehler bei Bild {img['id']} ({img['image_path']}): {e}")
            print(traceback.format_exc())
            conn.execute("UPDATE images SET status = 'error', alt_text = ? WHERE id = ?",
                         (f"Fehler bei der Analyse: {str(e)[:200]}", img["id"]))

        processed += 1
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
        """SELECT i.* FROM images i
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
            None, generate_alt_text, img["image_path"], img["context_text"], effective_type,
            regen_width, regen_height, regen_original_alt, True  # force_regenerate=True
        )

        langbeschreibung = result.get("langbeschreibung", "")
        conn.execute(
            """UPDATE images SET alt_text = ?, image_type = ?, konfidenz = ?,
               langbeschreibung = ?, alt_text_edited = NULL,
               needs_review = ?, pipeline_steps = ?, validation_result = ?, status = 'done' WHERE id = ?""",
            (_append_link_reference(result["alt_text"], img["context_text"] or ""), result["bildtyp"], result.get("konfidenz", "mittel"),
             langbeschreibung,
             1 if result.get("needs_review") else 0,
             result.get("pipeline_steps", ""),
             result.get("validation_result", ""),
             image_id)
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
        conn.execute("UPDATE images SET status = 'done' WHERE id = ?", (image_id,))
        conn.commit()
        conn.close()
        raise HTTPException(status_code=500, detail=f"Fehler bei der Neugenerierung: {str(e)}")


# ─── Export Routes ───────────────────────────────────────────

@app.post("/api/projects/{project_id}/export")
async def export_pdf(project_id: int, user: dict = Depends(get_current_user)):
    conn = get_db()
    project = conn.execute(
        "SELECT * FROM projects WHERE id = ? AND user_id = ?", (project_id, user["id"])
    ).fetchone()
    if not project:
        conn.close()
        raise HTTPException(status_code=404, detail="Projekt nicht gefunden")

    # PDF export only works for PDF projects
    project_type = project["project_type"] if "project_type" in project.keys() else "pdf"
    if project_type != "pdf":
        conn.close()
        raise HTTPException(status_code=400, detail="PDF-Export ist nur fuer PDF-Projekte verfuegbar")

    images = conn.execute(
        "SELECT * FROM images WHERE project_id = ?", (project_id,)
    ).fetchall()
    conn.close()

    # PDFIX-INTEGRATION (24.04.2026): getaggte PDFs ueber Heines Import-Script,
    # alle anderen wie bisher ueber write_alt_texts_to_pdf().
    extraction_method = project["extraction_method"] if "extraction_method" in project.keys() else "fitz"

    alt_texts = {}
    alt_texts_by_lfnr = {}
    image_metadata = []

    for img in images:
        alt_text = _display_alt_text(img)
        if alt_text is not None and img["xref"]:
            alt_texts[img["xref"]] = alt_text
        if alt_text is not None and img["image_index"]:
            alt_texts_by_lfnr[int(img["image_index"])] = alt_text

        # Build metadata for vector graphics support
        bbox = None
        if img["bbox_x0"] is not None:
            bbox = (img["bbox_x0"], img["bbox_y0"], img["bbox_x1"], img["bbox_y1"])

        image_metadata.append({
            "xref": img["xref"],
            "page_number": img["page_number"],
            "is_vector": bool(img["is_vector"]) if img["is_vector"] is not None else False,
            "bbox": bbox,
            "alt_text": alt_text,
            "image_path": img["image_path"],
        })

    output_dir = os.path.join(RESULTS_DIR, str(user["id"]), str(project_id))
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"inkludocs_{project['filename']}")

    headers = {}
    result = None

    if extraction_method == "pdfix":
        # PDFIX-INTEGRATION: Heines Import-Script ueber subprocess
        try:
            import pdfix_roundtrip as _pdfix
            count = _pdfix.import_alt_texts_pdfix(
                project["original_path"], output_path,
                alt_texts_by_lfnr, work_dir=output_dir)
            headers["X-Export-Method"] = "pdfix"
            headers["X-Export-Tagged"] = str(count)
            headers["X-Export-Total"] = str(len(alt_texts_by_lfnr))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"PDFix-Export fehlgeschlagen: {str(e)}")
    else:
        try:
            from pdf_export import write_alt_texts_to_pdf
            result = write_alt_texts_to_pdf(project["original_path"], output_path, alt_texts, image_metadata)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Export fehlgeschlagen: {str(e)}")

        # Build response with export warnings in headers
        export_warnings = []
        if isinstance(result, dict):
            export_warnings = result.get("warnings", [])
            tagged_count = result.get("tagged_count", 0)
            total_count = len(alt_texts)
            headers["X-Export-Method"] = "fitz"
            headers["X-Export-Tagged"] = str(tagged_count)
            headers["X-Export-Total"] = str(total_count)
            if export_warnings:
                headers["X-Export-Warnings"] = json.dumps(export_warnings, ensure_ascii=False)

    return FileResponse(
        output_path,
        filename=f"inkludocs_{project['filename']}",
        media_type="application/pdf",
        headers=headers
    )


@app.post("/api/projects/{project_id}/export/json")
async def export_json(project_id: int, user: dict = Depends(get_current_user)):
    """Export all alt-texts as JSON."""
    conn = get_db()
    project = conn.execute(
        "SELECT * FROM projects WHERE id = ? AND user_id = ?", (project_id, user["id"])
    ).fetchone()
    if not project:
        conn.close()
        raise HTTPException(status_code=404, detail="Projekt nicht gefunden")

    images = conn.execute(
        "SELECT * FROM images WHERE project_id = ? ORDER BY page_number, image_index", (project_id,)
    ).fetchall()
    conn.close()

    export_data = {
        "projekt": project["filename"],
        "bilder": [],
    }

    for img in images:
        alt_text = _display_alt_text(img)
        entry = {
            "alt_text": alt_text or "",
        }
        langbeschreibung = img["langbeschreibung"] if img["langbeschreibung"] else ""
        if langbeschreibung:
            entry["langbeschreibung"] = langbeschreibung
        export_data["bilder"].append(entry)

    json_bytes = json.dumps(export_data, ensure_ascii=False, indent=2).encode("utf-8")
    return StreamingResponse(
        io.BytesIO(json_bytes),
        media_type="application/json",
        headers={"Content-Disposition": f'attachment; filename="inkludocs_{project["filename"]}.json"'}
    )


@app.post("/api/projects/{project_id}/export/csv")
async def export_csv(project_id: int, user: dict = Depends(get_current_user)):
    """Export all alt-texts as CSV."""
    conn = get_db()
    project = conn.execute(
        "SELECT * FROM projects WHERE id = ? AND user_id = ?", (project_id, user["id"])
    ).fetchone()
    if not project:
        conn.close()
        raise HTTPException(status_code=404, detail="Projekt nicht gefunden")

    images = conn.execute(
        "SELECT * FROM images WHERE project_id = ? ORDER BY page_number, image_index", (project_id,)
    ).fetchall()
    conn.close()

    output = io.StringIO()
    writer = csv.writer(output, delimiter=";")
    writer.writerow(["Alt-Text", "Langbeschreibung"])

    for img in images:
        alt_text = _display_alt_text(img)
        writer.writerow([
            alt_text or "",
            img["langbeschreibung"] or "",
        ])

    csv_bytes = output.getvalue().encode("utf-8-sig")  # BOM for Excel compatibility
    return StreamingResponse(
        io.BytesIO(csv_bytes),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="inkludocs_{project["filename"]}.csv"'}
    )


@app.post("/api/projects/{project_id}/export/xlsx")
async def export_xlsx(project_id: int, user: dict = Depends(get_current_user)):
    """Export alt-texts as Excel with embedded images."""
    from openpyxl import Workbook
    from openpyxl.drawing.image import Image as XlImage
    from openpyxl.styles import Font, Alignment
    from openpyxl.utils.units import pixels_to_EMU

    conn = get_db()
    project = conn.execute(
        "SELECT * FROM projects WHERE id = ? AND user_id = ?", (project_id, user["id"])
    ).fetchone()
    if not project:
        conn.close()
        raise HTTPException(status_code=404, detail="Projekt nicht gefunden")

    images = conn.execute(
        "SELECT * FROM images WHERE project_id = ? ORDER BY page_number, image_index", (project_id,)
    ).fetchall()
    conn.close()

    wb = Workbook()
    ws = wb.active
    ws.title = "Alt-Texte"

    # Header row
    ws["A1"] = "Bild"
    ws["B1"] = "Alt-Text"
    ws["C1"] = "Langbeschreibung"
    for cell in [ws["A1"], ws["B1"], ws["C1"]]:
        cell.font = Font(bold=True, size=12)
    ws.column_dimensions["A"].width = 25
    ws.column_dimensions["B"].width = 60
    ws.column_dimensions["C"].width = 60

    for i, img in enumerate(images):
        row = i + 2
        alt_text = _display_alt_text(img)
        langbeschreibung = img["langbeschreibung"] or ""

        # Image filename for screenreaders + embedded image for sighted users
        img_path = img["image_path"]
        img_filename = os.path.basename(img_path) if img_path else "unbekannt"
        ws[f"A{row}"] = img_filename
        ws[f"A{row}"].alignment = Alignment(vertical="top")

        if os.path.exists(img_path):
            try:
                # openpyxl cannot handle WebP/AVIF — convert to PNG in memory
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

    output = io.BytesIO()
    wb.save(output)
    output.seek(0)

    return StreamingResponse(
        output,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f'attachment; filename="inkludocs_{project["filename"]}.xlsx"'}
    )


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
        model_used = result.get("model_used", "mistral-small")

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
            "model_used": model_used,
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
        "model_used": result["model_used"],
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
  "model_used": "mistral-small",
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
<tr><td><code>model_used</code></td><td>Verwendetes KI-Modell</td></tr>
<tr><td><code>processing_time_ms</code></td><td>Verarbeitungszeit in Millisekunden</td></tr>
</tbody>
</table>

<h2 id="get-result">Ergebnis abrufen</h2>
<p><span class="badge badge-get">GET</span> <code>/api/v1/alt-text/{result_id}</code></p>
<p>Ruft ein gespeichertes Ergebnis anhand seiner <code>result_id</code> ab. Erfordert denselben <code>X-API-Key</code> wie beim Erstellen.</p>
<pre><code>curl %%BASE_URL%%/api/v1/alt-text/abc123xyz \\
  -H "X-API-Key: idocs_deinSchluessel"</code></pre>
<p>Response-Felder: <code>result_id</code>, <code>alt_text</code>, <code>langbeschreibung</code>, <code>bildtyp</code>, <code>konfidenz</code>, <code>model_used</code>, <code>created_at</code>, <code>updated_at</code></p>

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
<li><strong>60 Anfragen pro Minute</strong></li>
<li><strong>1.000 Anfragen pro Tag</strong></li>
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
    {"datum": "13.05.2026", "text": "Chat-Assistent neu: Jedes Projekt hat jetzt einen eigenen Chatbot, der Bilder einsieht, Alt-Texte bewertet, Vorschlaege macht und nach Bestaetigung direkt in den Text uebernimmt. Aktuell auf Deutsch."},
    {"datum": "13.05.2026", "text": "Mehrsprachigkeit in Vorbereitung: Englisch, Franzoesisch und Spanisch werden in den kommenden Wochen nach und nach ausgerollt."},
    {"datum": "14.04.2026", "text": "Neue dreistufige Pruefpipeline aktiv: Klassifikation, Generierung und automatische Qualitaetspruefung gegen Halluzinationen. Laufende Auswertung zur weiteren Verbesserung."},
    {"datum": "07.04.2026", "text": "Alt-Text-Qualitaet verbessert: Produktbilder, Diagramme und verlinkte Bilder werden besser erkannt"},
    {"datum": "07.04.2026", "text": "Neue Bildformate: AVIF und HEIC werden jetzt unterstuetzt"},
    {"datum": "06.04.2026", "text": "Tageslimit: 100 Bilder pro Tag – Anzeige im Header"},
    {"datum": "06.04.2026", "text": "Registrierung offen: Konto erstellen mit E-Mail-Bestaetigung"},
    {"datum": "06.04.2026", "text": "InkluDocs unterstuetzen: Freiwillige Beitraege per PayPal moeglich"},
]

@app.get("/api/news")
async def get_news():
    """Return changelog entries for the Neuigkeiten panel."""
    return {"news": NEUIGKEITEN}

# ─── Frontend Routes ─────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
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

@app.get("/impressum", response_class=HTMLResponse)
async def impressum_page():
    return open("/app/frontend/impressum.html").read()

@app.get("/datenschutz", response_class=HTMLResponse)
async def datenschutz_page():
    return open("/app/frontend/datenschutz.html").read()

@app.get("/nutzungsbedingungen", response_class=HTMLResponse)
async def nutzungsbedingungen_page():
    return open("/app/frontend/nutzungsbedingungen.html").read()
