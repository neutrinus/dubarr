import os
import logging
import asyncio
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect, Depends, UploadFile, File
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, RedirectResponse, FileResponse
from pydantic import BaseModel

from infrastructure.database import Database
from infrastructure.ffmpeg import FFmpegWrapper
from config import setup_logging, OUTPUT_FOLDER, API_USER, API_PASS, VIDEO_FOLDER, TARGET_LANGS, DB_PATH

logger = logging.getLogger(__name__)

# Authentication setup
security = HTTPBasic()


def authenticate(credentials: HTTPBasicCredentials = Depends(security)):
    import secrets

    is_correct_user = secrets.compare_digest(credentials.username.encode("utf8"), API_USER.encode("utf8"))
    is_correct_pass = secrets.compare_digest(credentials.password.encode("utf8"), API_PASS.encode("utf8"))
    if not (is_correct_user and is_correct_pass):
        raise HTTPException(status_code=401, detail="Incorrect username or password", headers={"WWW-Authenticate": "Basic"})
    return credentials.username


# Global objects
db = Database(DB_PATH)
worker_task = None
dubber_base = None

TEMPLATE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "templates")
templates = Jinja2Templates(directory=TEMPLATE_DIR)


class WebhookPayload(BaseModel):
    path: str
    eventType: Optional[str] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    logger.info("Lifespan: Starting application setup...")

    # Reset interrupted tasks
    count = db.reset_interrupted_tasks()
    if count > 0:
        logger.warning(f"Reset {count} interrupted tasks to QUEUED state.")

    # Initialize AIDubber components and start worker
    from main import AIDubber
    from core.worker import JobWorker
    from core.pipeline import DubbingPipeline
    import threading

    global dubber_base
    dubber_base = AIDubber()

    # Pre-load models in background
    logger.info("Lifespan: Starting background model pre-loading...")
    threading.Thread(target=dubber_base.llm_manager.load_model, daemon=True).start()
    threading.Thread(target=dubber_base.tts_manager.load_engine, daemon=True).start()

    def pipeline_factory():
        return DubbingPipeline(
            llm_manager=dubber_base.llm_manager,
            tts_manager=dubber_base.tts_manager,
            target_langs=dubber_base.target_langs,
            db=db,
            debug_mode=dubber_base.debug_mode,
        )

    global worker_task
    worker_task = JobWorker(db, pipeline_factory)
    worker_task.start()

    yield

    logger.info("Lifespan: Shutting down...")
    if worker_task:
        worker_task.stop()


app = FastAPI(lifespan=lifespan)


@app.get("/health")
async def health():
    return {
        "status": "online",
        "worker_alive": worker_task is not None and worker_task.is_alive(),
        "queue_stats": db.get_queue_stats(),
    }


@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    tasks = db.get_all_tasks()
    for t in tasks:
        t["progress"] = db.get_task_progress(t["id"])

    # Determine global system status
    sys_status = "Online"
    if dubber_base:
        ls = dubber_base.llm_manager.status
        ts = dubber_base.tts_manager.status
        if ls == "DOWNLOADING" or ts == "DOWNLOADING":
            sys_status = "Downloading Models"
        elif ls == "LOADING" or ts == "LOADING":
            sys_status = "Loading Engines"
        elif ls == "ERROR" or ts == "ERROR":
            sys_status = "Engine Error"

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "tasks": tasks,
            "worker_alive": True,
            "system_status": sys_status,
        },
    )


@app.post("/retry/{task_id}")
async def retry_task(task_id: int):
    db.retry_task(task_id)
    return RedirectResponse(url="/", status_code=303)


@app.post("/purge/{task_id}")
async def purge_task_cache(task_id: int):
    db.purge_task_cache(task_id)
    return RedirectResponse(url="/", status_code=303)


@app.post("/delete/{task_id}")
async def delete_task(task_id: int):
    db.delete_task(task_id)
    return RedirectResponse(url="/", status_code=303)


@app.get("/download/{task_id}")
async def download_task(task_id: int, username: str = Depends(authenticate)):
    tasks = db.get_all_tasks()
    task = next((t for t in tasks if t["id"] == task_id), None)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    file_path = task["path"]
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found on disk")

    return FileResponse(path=file_path, filename=os.path.basename(file_path), media_type="application/octet-stream")


@app.post("/upload")
async def upload_video(file: UploadFile = File(...), username: str = Depends(authenticate)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    # Save file to VIDEO_FOLDER
    target_path = os.path.join(VIDEO_FOLDER, file.filename)

    # Check for existing file
    if os.path.exists(target_path):
        # Could append suffix, but for now just overwrite or return error
        pass

    try:
        with open(target_path, "wb") as buffer:
            import shutil

            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        logger.error(f"Failed to save uploaded file: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")

    # Add task to DB
    try:
        meta_raw = FFmpegWrapper.get_metadata(target_path)
        duration = float(meta_raw.get("format", {}).get("duration", 0))
        audio_langs = [
            s.get("tags", {}).get("language", "und") for s in meta_raw.get("streams", []) if s.get("codec_type") == "audio"
        ]
        meta = {
            "size": os.path.getsize(target_path),
            "duration": duration,
            "source_lang": audio_langs[0] if audio_langs else "und",
            "has_subs": any(s.get("codec_type") == "subtitle" for s in meta_raw.get("streams", [])),
            "target_langs": ",".join(TARGET_LANGS),
        }
    except Exception:
        meta = {}

    db.add_task(target_path, meta)

    return RedirectResponse(url="/", status_code=303)


@app.post("/webhook")
async def receive_webhook(payload: WebhookPayload, username: str = Depends(authenticate)):
    if not payload.path:
        raise HTTPException(status_code=400, detail="Path is required")

    full_path = (
        payload.path if os.path.isabs(payload.path) else os.path.join(VIDEO_FOLDER, payload.path.replace("videos/", ""))
    )
    full_path = os.path.normpath(full_path)

    # Get metadata
    try:
        meta_raw = FFmpegWrapper.get_metadata(full_path)
        duration = float(meta_raw.get("format", {}).get("duration", 0))
        audio_langs = [
            s.get("tags", {}).get("language", "und") for s in meta_raw.get("streams", []) if s.get("codec_type") == "audio"
        ]
        meta = {
            "size": os.path.getsize(full_path),
            "duration": duration,
            "source_lang": audio_langs[0] if audio_langs else "und",
            "has_subs": any(s.get("codec_type") == "subtitle" for s in meta_raw.get("streams", [])),
            "target_langs": ",".join(TARGET_LANGS),
        }
    except Exception:
        meta = {}

    status = db.add_task(full_path, meta)
    return {"status": status, "path": full_path}


@app.websocket("/ws/logs")
async def websocket_logs(websocket: WebSocket):
    await websocket.accept()
    log_file = os.path.join(OUTPUT_FOLDER, "processing.log")
    if not os.path.exists(log_file):
        with open(log_file, "w") as f:
            f.write("Log file created.\n")
    try:
        with open(log_file, "r") as f:
            lines = f.readlines()[-50:]
            for line in lines:
                await websocket.send_text(line.strip())
            f.seek(0, os.SEEK_END)
            while True:
                line = f.readline()
                if not line:
                    await asyncio.sleep(0.5)
                    continue
                await websocket.send_text(line.strip())
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.error(f"WS error: {e}")
