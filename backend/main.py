import os
import shutil
import json
import asyncio
import secrets
import time
from collections import defaultdict
from datetime import datetime, timedelta
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, Request
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from jose import jwt, JWTError

from database import (
    init_db, get_db, create_user, verify_user, get_user_by_email, get_user_by_id,
    create_password_reset_token, verify_reset_token, reset_password,
    list_all_users, update_user_active, delete_user_data, admin_reset_password,
)
from pdf_processor import extract_images_from_pdf, generate_alt_text

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
    allow_headers=["Content-Type"],
    allow_credentials=True,
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

    # Since we don't have email sending yet, return the link directly
    # In production, this would send an email
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
async def upload_pdf(file: UploadFile = File(...), user: dict = Depends(get_current_user)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Nur PDF-Dateien erlaubt")

    # Read and check file size
    content = await file.read()
    if len(content) > MAX_UPLOAD_SIZE:
        raise HTTPException(status_code=413, detail=f"Datei zu gross. Maximum: {MAX_UPLOAD_SIZE // (1024*1024)} MB")

    # Create user directory
    user_dir = os.path.join(UPLOAD_DIR, str(user["id"]))
    os.makedirs(user_dir, exist_ok=True)

    # Save file with sanitized name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = f"{timestamp}_{os.path.basename(file.filename)}"
    file_path = os.path.join(user_dir, safe_name)
    with open(file_path, "wb") as f:
        f.write(content)

    # Create project in DB
    conn = get_db()
    cursor = conn.execute(
        "INSERT INTO projects (user_id, filename, original_path, status) VALUES (?, ?, ?, 'extracting')",
        (user["id"], file.filename, file_path)
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

    # Store images in DB
    for img in images:
        conn.execute(
            """INSERT INTO images (project_id, page_number, image_index, image_path, context_text, width, height, xref)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (project_id, img["page_number"], img["image_index"], img["image_path"],
             img["context_text"], img["width"], img["height"], img["xref"])
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
        "filename": file.filename,
        "total_images": len(images),
    }


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
    if os.path.exists(project["original_path"]):
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
            None, generate_alt_text, img["image_path"], img["context_text"]
        )

        conn.execute(
            """UPDATE images SET alt_text = ?, image_type = ?, konfidenz = ?, status = 'done' WHERE id = ?""",
            (result["alt_text"], result["bildtyp"], result.get("konfidenz", "mittel"), img["id"])
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
    conn.commit()
    conn.close()
    return {"ok": True}


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
    for img in images:
        alt_text = img["alt_text_edited"] if img["alt_text_edited"] else img["alt_text"]
        if alt_text is not None and img["xref"]:
            alt_texts[img["xref"]] = alt_text

    output_dir = os.path.join(RESULTS_DIR, str(user["id"]), str(project_id))
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"inkludocs_{project['filename']}")

    try:
        from pdf_processor import write_alt_texts_to_pdf
        write_alt_texts_to_pdf(project["original_path"], output_path, alt_texts)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Export fehlgeschlagen: {str(e)}")

    return FileResponse(
        output_path,
        filename=f"inkludocs_{project['filename']}",
        media_type="application/pdf"
    )


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
