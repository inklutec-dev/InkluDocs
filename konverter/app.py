"""InkluDocs Umwandler-Dienst (29.08.2026, Steve + Fable 5).

POST /pdfua  (multipart: datei=<.docx>)  ->  JSON
    {"ok": true, "pdf_b64": "...", "dauer_s": 4.2,
     "verapdf": {"compliant": bool, "profile": "PDF/UA-1 ...", "rules": [
         {"clause": "7.1", "test": 3, "description": "...", "failed": 17}, ...]}}
GET  /health  ->  {"ok": true, "soffice": "...", "verapdf": true}

LibreOffice wird headless mit eigenem Profil je Lauf gestartet (parallele Laeufe
kommen sich so nicht in die Quere), Exportfilter mit UseTaggedPDF + PDFUACompliance
(getaggte PDF mit Struktur, Lesezeichen, Sprache, Alt-Texten). veraPDF prueft
danach gegen das Profil PDF/UA-1 (ISO 14289-1). Der Dienst schreibt nichts
Dauerhaftes: jeder Lauf bekommt ein Temp-Verzeichnis, das am Ende geloescht wird.
"""
import asyncio, base64, glob, json, os, re, shutil, subprocess, tempfile, time, uuid

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

app = FastAPI(title="InkluDocs Umwandler", docs_url=None, redoc_url=None)

SOFFICE_TIMEOUT = int(os.environ.get("SOFFICE_TIMEOUT", "150"))
VERAPDF_TIMEOUT = int(os.environ.get("VERAPDF_TIMEOUT", "120"))
MAX_BYTES = int(os.environ.get("MAX_UPLOAD_BYTES", str(60 * 1024 * 1024)))
# LibreOffice ist nicht fuer viele parallele Instanzen gedacht: zwei gleichzeitig reichen.
_sem = asyncio.Semaphore(int(os.environ.get("PARALLEL", "2")))

PDF_FILTER = ('pdf:writer_pdf_Export:{'
              '"UseTaggedPDF":{"type":"boolean","value":"true"},'
              '"PDFUACompliance":{"type":"boolean","value":"true"},'
              '"ExportBookmarks":{"type":"boolean","value":"true"},'
              '"ExportNotes":{"type":"boolean","value":"false"},'
              '"SelectPdfVersion":{"type":"long","value":"17"}'
              '}')


def _verapdf_cmd(pdf_path: str) -> list:
    """veraPDF-Aufruf: Launcher-Skript aus dem Image; Java kommt von Debian."""
    launcher = "/opt/verapdf/verapdf"
    if os.path.exists(launcher):
        return ["sh", launcher, "-f", "ua1", "--format", "json", pdf_path]
    return ["java", "-cp", "/opt/verapdf/bin/*", "org.verapdf.apps.GreenfieldCliWrapper",
            "-f", "ua1", "--format", "json", pdf_path]


_VERAPDF_VERSION = None


def _verapdf_version() -> str:
    """Version des Pruefers, einmal ermittelt und gemerkt (Steve/Cody 30.08.2026).

    Steht in /health, damit bei einem strittigen Urteil sofort sichtbar ist, WELCHER
    Pruefer es gefaellt hat. Auf dem Server liegt ausserdem ein eigenstaendiges veraPDF
    (Handwerkszeug, andere Version) — massgeblich ist allein das hier im Umwandler.
    Der Aufruf startet eine JVM, deshalb nur beim ersten Mal."""
    global _VERAPDF_VERSION
    if _VERAPDF_VERSION is None:
        try:
            launcher = "/opt/verapdf/verapdf"
            cmd = (["sh", launcher, "--version"] if os.path.exists(launcher)
                   else ["java", "-cp", "/opt/verapdf/bin/*",
                         "org.verapdf.apps.GreenfieldCliWrapper", "--version"])
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=60).stdout or ""
            treffer = re.search(r"veraPDF\s+(\S+)", out)
            _VERAPDF_VERSION = treffer.group(1) if treffer else "unbekannt"
        except Exception as e:  # noqa: BLE001
            _VERAPDF_VERSION = "unbekannt (%s)" % type(e).__name__
    return _VERAPDF_VERSION


def _verapdf(pdf_path: str) -> dict:
    p = subprocess.run(_verapdf_cmd(pdf_path), capture_output=True, text=True, timeout=VERAPDF_TIMEOUT)
    out = p.stdout.strip()
    if not out:
        raise RuntimeError(f"veraPDF ohne Ausgabe (rc={p.returncode}): {p.stderr[-400:]}")
    d = json.loads(out)
    job = d["report"]["jobs"][0]
    vr = job.get("validationResult")
    if isinstance(vr, list):
        vr = vr[0]
    if not vr:
        raise RuntimeError("veraPDF: kein validationResult (Datei nicht prüfbar?)")
    det = vr.get("details") or {}
    rules = []
    for r in det.get("ruleSummaries", []) or []:
        if r.get("ruleStatus") != "FAILED":
            continue
        rules.append({"clause": str(r.get("clause") or ""), "test": r.get("testNumber"),
                      "description": r.get("description") or "", "failed": int(r.get("failedChecks") or 0)})
    rules.sort(key=lambda x: (tuple(int(t) if t.isdigit() else 0 for t in x["clause"].split(".")), x["test"] or 0))
    return {"compliant": bool(vr.get("compliant")), "profile": vr.get("profileName") or "PDF/UA-1",
            "passed_checks": det.get("passedChecks"), "failed_checks": det.get("failedChecks"),
            "rules": rules}


def _konvertiere_sync(docx_path: str, work: str) -> str:
    profile = os.path.join(work, "lo_profile")
    env = dict(os.environ, HOME=work)
    cmd = ["soffice", f"-env:UserInstallation=file://{profile}", "--headless", "--norestore", "--nologo",
           "--nodefault", "--nolockcheck", "--convert-to", PDF_FILTER, "--outdir", work, docx_path]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=SOFFICE_TIMEOUT, env=env, cwd=work)
    pdfs = glob.glob(os.path.join(work, "*.pdf"))
    if not pdfs:
        raise RuntimeError(f"LibreOffice lieferte keine PDF (rc={p.returncode}): {(p.stderr or p.stdout)[-400:]}")
    return pdfs[0]


@app.get("/health")
async def health():
    try:
        v = subprocess.run(["soffice", "--version"], capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception as e:  # noqa: BLE001
        v = f"fehlt: {e}"
    return {"ok": True, "soffice": v, "verapdf": os.path.exists("/opt/verapdf/bin"),
            "verapdf_version": _verapdf_version()}


@app.post("/pruefe")
async def pruefe(datei: UploadFile = File(...)):
    """Nur veraPDF (PDF/UA-1) fuer eine fertige PDF — z. B. nach der Nachbearbeitung in der App."""
    data = await datei.read()
    if len(data) > MAX_BYTES:
        raise HTTPException(status_code=413, detail="Datei zu gross")
    if data[:5] != b"%PDF-":
        raise HTTPException(status_code=400, detail="Keine PDF")
    t0 = time.time()
    work = tempfile.mkdtemp(prefix="p_", dir="/work")
    try:
        pfad = os.path.join(work, "pruefling.pdf")
        with open(pfad, "wb") as f:
            f.write(data)
        async with _sem:
            report = await asyncio.get_running_loop().run_in_executor(None, _verapdf, pfad)
        return JSONResponse({"ok": True, "dauer_s": round(time.time() - t0, 1), "verapdf": report})
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=504, detail="Pruefung hat zu lange gedauert")
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Pruefung fehlgeschlagen: {e}"[:500])
    finally:
        shutil.rmtree(work, ignore_errors=True)


@app.post("/pdfua")
async def pdfua(datei: UploadFile = File(...)):
    name = os.path.basename(datei.filename or "dokument.docx")
    if not name.lower().endswith(".docx"):
        raise HTTPException(status_code=400, detail="Nur .docx wird umgewandelt")
    data = await datei.read()
    if len(data) > MAX_BYTES:
        raise HTTPException(status_code=413, detail="Datei zu gross")
    t0 = time.time()
    work = tempfile.mkdtemp(prefix="k_", dir="/work")
    try:
        safe = re.sub(r"[^A-Za-z0-9._-]+", "_", name)[:120] or "dokument.docx"
        src = os.path.join(work, safe)
        with open(src, "wb") as f:
            f.write(data)
        async with _sem:
            loop = asyncio.get_running_loop()
            pdf_path = await loop.run_in_executor(None, _konvertiere_sync, src, work)
            report = await loop.run_in_executor(None, _verapdf, pdf_path)
        with open(pdf_path, "rb") as f:
            pdf_b64 = base64.b64encode(f.read()).decode("ascii")
        return JSONResponse({"ok": True, "pdf_b64": pdf_b64, "dauer_s": round(time.time() - t0, 1),
                             "verapdf": report, "id": uuid.uuid4().hex[:8]})
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=504, detail="Umwandlung hat zu lange gedauert")
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Umwandlung fehlgeschlagen: {e}"[:500])
    finally:
        shutil.rmtree(work, ignore_errors=True)
