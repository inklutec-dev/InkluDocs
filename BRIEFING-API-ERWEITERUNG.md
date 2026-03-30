# Briefing: InkluDocs API-Erweiterung – result_id & PATCH-Endpoint

## Ziel

Den bestehenden API-Endpoint `POST /api/v1/alt-text` erweitern, sodass jedes Ergebnis eine `result_id` zurückgibt. Einen neuen `PATCH /api/v1/alt-text/{result_id}` Endpoint hinzufügen, über den API-Nutzer den Alt-Text und die Langbeschreibung nachträglich ändern können – ohne sich in die Web-Oberfläche einloggen zu müssen.

## Hintergrund

Externe Nutzer (z.B. CMS-Plugin-Entwickler, PDF-Tools) integrieren unsere API. Die bekommen Alt-Text + Langbeschreibung zurück und zeigen das in ihrem eigenen Tool an. Wenn der Text nicht passt, sollen sie ihn direkt über die API korrigieren können. "Neu generieren" braucht keinen eigenen Endpoint – dafür schickt man das Bild einfach nochmal an POST.

## Ist-Zustand

### Bestehender Endpoint: `POST /api/v1/alt-text`
- Datei: `backend/main.py`, ab Zeile 1603
- Auth: `X-API-Key` Header, verifiziert über `get_api_user()`
- Input: Multipart (file/image) oder JSON (image_base64), optional context, language, image_type
- Output: `{ alt_text, langbeschreibung, bildtyp, konfidenz, model_used, processing_time_ms }`
- Bilder werden temporär gespeichert und nach Generierung gelöscht
- Usage wird in `api_usage` Tabelle geloggt

### Bestehende DB-Tabellen (relevant):
- `api_keys`: id, user_id, key_hash, name, created_at, last_used, is_active
- `api_usage`: id, api_key_id, user_id, timestamp, processing_time_ms, model_used, image_size_bytes, success, error_message

## Aufgabe

### 1. Neue DB-Tabelle: `api_results`

In `database.py` in der `init_db()` Funktion (im CREATE-Block) hinzufügen:

```sql
CREATE TABLE IF NOT EXISTS api_results (
    id TEXT PRIMARY KEY,              -- UUID/Token als result_id
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
```

### 2. Neue DB-Funktionen in `database.py`

```python
def create_api_result(result_id: str, user_id: int, api_key_id: int,
                      alt_text: str, langbeschreibung: str, bildtyp: str = "unbekannt",
                      konfidenz: str = "mittel", model_used: str = "",
                      processing_time_ms: int = 0, language: str = "de",
                      context_text: str = "") -> str:
    """Speichert ein API-Ergebnis und gibt die result_id zurück."""
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


def get_api_result(result_id: str, user_id: int) -> dict | None:
    """Holt ein API-Ergebnis. Prüft dass es dem User gehört."""
    conn = get_db()
    row = conn.execute(
        "SELECT * FROM api_results WHERE id = ? AND user_id = ?",
        (result_id, user_id)
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def update_api_result(result_id: str, user_id: int,
                      alt_text: str | None = None,
                      langbeschreibung: str | None = None) -> bool:
    """Aktualisiert Alt-Text und/oder Langbeschreibung eines API-Ergebnisses."""
    conn = get_db()
    # Prüfen ob das Ergebnis dem User gehört
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
        return True  # Nichts zu ändern

    updates.append("updated_at = datetime('now')")
    params.extend([result_id, user_id])

    conn.execute(
        f"UPDATE api_results SET {', '.join(updates)} WHERE id = ? AND user_id = ?",
        params
    )
    conn.commit()
    conn.close()
    return True
```

Diese Funktionen in den Import-Block von `main.py` aufnehmen:
```python
from database import (..., create_api_result, get_api_result, update_api_result)
```

### 3. POST-Endpoint erweitern (`/api/v1/alt-text`)

In `main.py`, im bestehenden `api_generate_alt_text()` – nach der erfolgreichen Generierung (dort wo `response_data` zusammengebaut wird):

**Vor** dem `return JSONResponse(...)`:

```python
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
```

Und `result_id` zum `response_data` dict hinzufügen:
```python
response_data["result_id"] = result_id
```

### 4. Neuer PATCH-Endpoint

Direkt nach dem bestehenden POST-Endpoint einfügen:

```python
@app.patch("/api/v1/alt-text/{result_id}")
async def api_update_alt_text(result_id: str, request: Request):
    """Aktualisiert den Alt-Text und/oder die Langbeschreibung eines API-Ergebnisses.
    Erfordert X-API-Key Header. Nur der Besitzer kann sein Ergebnis ändern."""
    api_user = get_api_user(request)

    try:
        data = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Ungültiges JSON")

    alt_text = data.get("alt_text")
    langbeschreibung = data.get("langbeschreibung")

    if alt_text is None and langbeschreibung is None:
        raise HTTPException(
            status_code=400,
            detail="Mindestens 'alt_text' oder 'langbeschreibung' muss angegeben werden"
        )

    # Validierung: Texte dürfen nicht leer sein wenn angegeben
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

    # Aktualisiertes Ergebnis zurückgeben
    updated = get_api_result(result_id, api_user["id"])
    return JSONResponse(content={
        "result_id": updated["id"],
        "alt_text": updated["alt_text"],
        "langbeschreibung": updated["langbeschreibung"],
        "bildtyp": updated["bildtyp"],
        "updated_at": updated["updated_at"],
    })
```

### 5. Neuer GET-Endpoint (optional, aber sinnvoll)

Damit API-Nutzer ein gespeichertes Ergebnis abrufen können:

```python
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
```

### 6. API-Doku aktualisieren

Den bestehenden `/api/v1/docs` Endpoint (ab Zeile 1747 in main.py) erweitern. Zu den vorhandenen Beispielen die neuen Endpoints und das `result_id` Feld in der Response dokumentieren. Mindestens:

- POST Response enthält jetzt `result_id`
- PATCH `/api/v1/alt-text/{result_id}` mit Beispiel
- GET `/api/v1/alt-text/{result_id}` mit Beispiel

## WICHTIG: Was NICHT gemacht werden soll

- KEIN separater Regenerate-Endpoint – einfach Bild nochmal an POST schicken
- KEINE automatische Projekt-Erstellung (kommt später in Phase 2)
- KEIN Credit-System ändern – bestehende Usage-Logik bleibt wie sie ist
- KEINE Frontend-Änderungen nötig – das sind rein Backend/API-Änderungen
- Bestehende Endpoints NICHT verändern (außer dem POST wie beschrieben)

## Zielumgebung

- Codebase: `/home/openclaw/.openclaw/workspace/InkluDocs/`
- Backend: `backend/main.py` und `backend/database.py`
- Python/FastAPI
- Datenbank: SQLite
- Deploy: NUR auf Staging! (`docker-compose.staging.yml`)
  - Build: `cd /home/openclaw/.openclaw/workspace/InkluDocs && docker compose -f docker-compose.staging.yml build && docker compose -f docker-compose.staging.yml up -d`
  - Staging-URL: https://staging.inkludocs.inklutec.de
  - NICHT auf Produktion deployen!

## Testen

Nach dem Deploy auf Staging folgende Tests manuell durchführen:

1. POST `/api/v1/alt-text` mit einem Bild → prüfen ob `result_id` im Response ist
2. GET `/api/v1/alt-text/{result_id}` → gespeichertes Ergebnis abrufen
3. PATCH `/api/v1/alt-text/{result_id}` mit `{"alt_text": "Neuer Text"}` → prüfen ob geändert
4. PATCH mit fremdem API-Key → muss 404 geben (Sicherheit)
5. API-Doku unter `/api/v1/docs` prüfen ob die neuen Endpoints dokumentiert sind
