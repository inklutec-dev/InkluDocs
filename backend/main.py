import os
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
    create_api_key, verify_api_key, list_api_keys, delete_api_key,
)
from pdf_processor import extract_images_from_pdf, generate_alt_text, generate_alt_text_for_image

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

# Allowed image extensions for direct upload
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".svg", ".webp", ".heic", ".heif", ".bmp", ".tiff", ".tif"}

# Rate limiting for login
_login_attempts = defaultdict(list)
MAX_LOGIN_ATTEMPTS = 5
LOGIN_WINDOW_SECONDS = 300  # 5 minutes


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    # Create default admin user if not exists
    try:
        create_user("kontakt@inklutec.de", "inkludocs2025", "Administrator", is_admin=1)
        print("Default admin user created")
    except Exception:
        pass
    yield


app = FastAPI(title="InkluDocs", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://inkludocs.inklutec.de"],
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["Content-Type", "X-API-Key"],
    allow_credentials=True,
    expose_headers=["X-Export-Warnings", "X-Export-Tagged", "X-Export-Total"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="/app/frontend"), name="static")


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
    return {"id": key_info["user_id"], "email": key_info["email"], "is_admin": 0}


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
    except Exception as e:
        raise HTTPException(status_code=500, detail="Registrierung fehlgeschlagen")

    token = create_token(user_id, email, 0)
    response = JSONResponse({
        "ok": True,
        "email": email,
        "display_name": display_name,
        "is_admin": 0,
    })
    response.set_cookie("token", token, httponly=True, samesite="strict", max_age=TOKEN_EXPIRE_HOURS * 3600)
    return response


@app.post("/api/logout")
async def logout():
    response = JSONResponse({"ok": True})
    response.delete_cookie("token")
    return response


@app.get("/api/me")
async def me(user: dict = Depends(get_current_user)):
    db_user = get_user_by_id(user["id"])
    if not db_user:
        raise HTTPException(status_code=401, detail="User nicht gefunden")
    return {
        "ok": True,
        "user": {
            "id": db_user["id"],
            "email": db_user["email"],
            "display_name": db_user["display_name"],
            "is_admin": db_user["is_admin"],
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

    return {
        "ok": True,
        "message": "Reset-Link wurde erstellt.",
        "reset_url": reset_url,
        "hinweis": "Da kein E-Mail-Versand konfiguriert ist, wird der Link hier direkt angezeigt. Als Admin koennen Sie auch Passwoerter direkt zuruecksetzen.",
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
               width, height, xref, bbox_x0, bbox_y0, bbox_x1, bbox_y1, is_vector)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (project_id, img["page_number"], img["image_index"], img["image_path"],
             img["context_text"], img["width"], img["height"], img["xref"],
             bbox_x0, bbox_y0, bbox_x1, bbox_y1, is_vector)
        )

    conn.execute(
        "UPDATE projects SET status = 'extracted', total_images = ? WHERE id = ?",
        (len(images), project_id)
    )
    conn.commit()
    conn.close()

    return {
        "ok": True,
        "project_id": project_id,
        "filename": filename,
        "total_images": len(images),
        "project_type": "pdf",
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
                "User-Agent": "InkluDocs/1.0 (Barrierefreiheits-Scanner; kontakt@inklutec.de)"
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
    async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as client:
        for idx, img_tag in enumerate(img_tags, 1):
            src = img_tag.get("src", "")
            if not src:
                continue

            # Resolve relative URLs
            img_url = urljoin(url, src)

            # Get original alt text
            original_alt = img_tag.get("alt", "")

            # Get context: parent text, figcaption, title attribute
            context_parts = []
            if img_tag.get("title"):
                context_parts.append(f"[title] {img_tag['title']}")
            if original_alt:
                context_parts.append(f"[Original alt] {original_alt}")
            # figcaption
            parent_figure = img_tag.find_parent("figure")
            if parent_figure:
                figcaption = parent_figure.find("figcaption")
                if figcaption:
                    context_parts.append(f"[Bildunterschrift] {figcaption.get_text(strip=True)}")
            # Surrounding text (parent element)
            parent = img_tag.parent
            if parent and parent.name not in ("html", "body", "head"):
                parent_text = parent.get_text(strip=True)[:200]
                if parent_text and parent_text != original_alt:
                    context_parts.append(f"[Umgebungstext] {parent_text}")
            context_text = "\n".join(context_parts) if context_parts else ""

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


def _ext_from_content_type(ct: str) -> str:
    """Map content type to file extension."""
    ct = ct.lower().split(";")[0].strip()
    mapping = {
        "image/jpeg": ".jpg",
        "image/png": ".png",
        "image/gif": ".gif",
        "image/webp": ".webp",
        "image/svg+xml": ".svg",
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


@app.post("/api/projects/{project_id}/generate")
async def generate_alt_texts(project_id: int, user: dict = Depends(get_current_user)):
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
    conn = get_db()
    images = conn.execute(
        "SELECT * FROM images WHERE project_id = ? AND status = 'pending' ORDER BY page_number, image_index",
        (project_id,)
    ).fetchall()

    processed = 0
    for img in images:
        conn.execute("UPDATE images SET status = 'processing' WHERE id = ?", (img["id"],))
        conn.commit()

        result = await asyncio.get_event_loop().run_in_executor(
            None, generate_alt_text, img["image_path"], img["context_text"], None
        )

        langbeschreibung = result.get("langbeschreibung", "")
        conn.execute(
            """UPDATE images SET alt_text = ?, image_type = ?, konfidenz = ?, langbeschreibung = ?, status = 'done' WHERE id = ?""",
            (result["alt_text"], result["bildtyp"], result.get("konfidenz", "mittel"), langbeschreibung, img["id"])
        )
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


# ─── Regenerate Single Image ────────────────────────────────

@app.post("/api/projects/{project_id}/regenerate/{image_id}")
async def regenerate_image(project_id: int, image_id: int, request: Request, user: dict = Depends(get_current_user)):
    """Regenerate alt-text for a single image with optional specialized prompt."""
    data = await request.json()
    image_type = data.get("image_type")  # Optional: foto, diagramm, karte, etc.

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
        result = await asyncio.get_event_loop().run_in_executor(
            None, generate_alt_text, img["image_path"], img["context_text"], image_type
        )

        langbeschreibung = result.get("langbeschreibung", "")
        conn.execute(
            """UPDATE images SET alt_text = ?, image_type = ?, konfidenz = ?,
               langbeschreibung = ?, alt_text_edited = NULL, status = 'done' WHERE id = ?""",
            (result["alt_text"], result["bildtyp"], result.get("konfidenz", "mittel"),
             langbeschreibung, image_id)
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

    images = conn.execute(
        "SELECT * FROM images WHERE project_id = ?", (project_id,)
    ).fetchall()
    conn.close()

    alt_texts = {}
    image_metadata = []

    for img in images:
        alt_text = img["alt_text_edited"] if img["alt_text_edited"] else img["alt_text"]
        if alt_text is not None and img["xref"]:
            alt_texts[img["xref"]] = alt_text

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

    try:
        from pdf_export import write_alt_texts_to_pdf
        result = write_alt_texts_to_pdf(project["original_path"], output_path, alt_texts, image_metadata)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Export fehlgeschlagen: {str(e)}")

    # Build response with export warnings in headers
    headers = {}
    export_warnings = []
    if isinstance(result, dict):
        export_warnings = result.get("warnings", [])
        tagged_count = result.get("tagged_count", 0)
        total_count = len(alt_texts)
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
        "erstellt": datetime.now().isoformat(),
        "quelle": dict(project).get("source_url", ""),
        "bilder": [],
    }

    for img in images:
        alt_text = img["alt_text_edited"] if img["alt_text_edited"] else img["alt_text"]
        entry = {
            "id": img["id"],
            "seite": img["page_number"],
            "index": img["image_index"],
            "bildtyp": img["image_type"],
            "alt_text": alt_text or "",
            "breite": img["width"],
            "hoehe": img["height"],
        }
        langbeschreibung = img["langbeschreibung"] if img["langbeschreibung"] else ""
        if langbeschreibung:
            entry["langbeschreibung"] = langbeschreibung
        original_alt = img["original_alt"] if img["original_alt"] else ""
        if original_alt:
            entry["original_alt"] = original_alt
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
    writer.writerow(["ID", "Seite", "Index", "Bildtyp", "Alt-Text", "Langbeschreibung", "Original-Alt", "Breite", "Hoehe"])

    for img in images:
        alt_text = img["alt_text_edited"] if img["alt_text_edited"] else img["alt_text"]
        writer.writerow([
            img["id"],
            img["page_number"],
            img["image_index"],
            img["image_type"],
            alt_text or "",
            img["langbeschreibung"] or "",
            img["original_alt"] or "",
            img["width"],
            img["height"],
        ])

    csv_bytes = output.getvalue().encode("utf-8-sig")  # BOM for Excel compatibility
    return StreamingResponse(
        io.BytesIO(csv_bytes),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="inkludocs_{project["filename"]}.csv"'}
    )


# ─── Public API ──────────────────────────────────────────────

@app.post("/v1/alt-text")
async def api_generate_alt_text(request: Request):
    """Public API endpoint for alt-text generation. Requires X-API-Key header."""
    api_user = get_api_user(request)

    # Parse multipart form data
    form = await request.form()
    file = form.get("file")
    if not file:
        raise HTTPException(status_code=400, detail="Bitte eine Bilddatei als 'file' hochladen")

    context_text = form.get("context", "")
    image_type = form.get("image_type")

    # Validate file
    filename = file.filename or "image.jpg"
    ext = os.path.splitext(filename)[1].lower()
    if ext not in IMAGE_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Nur Bilddateien erlaubt (JPG, PNG, GIF, SVG, WebP)"
        )

    content = await file.read()
    if len(content) > MAX_UPLOAD_SIZE:
        raise HTTPException(status_code=413, detail="Datei zu gross")

    # Save temporarily
    tmp_dir = os.path.join(UPLOAD_DIR, "api_tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    tmp_path = os.path.join(tmp_dir, f"{secrets.token_hex(8)}{ext}")
    with open(tmp_path, "wb") as f:
        f.write(content)

    try:
        result = await asyncio.get_event_loop().run_in_executor(
            None, generate_alt_text_for_image, tmp_path, context_text, image_type
        )
        return {
            "alt_text": result.get("alt_text", ""),
            "bildtyp": result.get("bildtyp", "unbekannt"),
            "konfidenz": result.get("konfidenz", "mittel"),
            "langbeschreibung": result.get("langbeschreibung", ""),
            "ist_dekorativ": result.get("ist_dekorativ", False),
        }
    finally:
        # Clean up temporary file
        try:
            os.remove(tmp_path)
        except Exception:
            pass


# ─── Frontend Routes ─────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index():
    return open("/app/frontend/index.html").read()

@app.get("/register", response_class=HTMLResponse)
async def register_page():
    return open("/app/frontend/register.html").read()

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
    return open("/app/frontend/app.html").read()
