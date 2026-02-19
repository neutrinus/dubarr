import os
import logging
import asyncio
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect, Depends
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, RedirectResponse
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

    # Initialize AIDubber and start worker
    from main import AIDubber  # Temporary import

    dubber = AIDubber()

    # We need to bridge threading and asyncio for now
    import threading

    class LegacyWorkerWrapper(threading.Thread):
        def __init__(self, dubber, db):
            super().__init__(daemon=True)
            self.dubber = dubber
            self.db = db
            self.stop_signal = threading.Event()

        def run(self):
            while not self.stop_signal.is_set():
                task = self.db.fetch_next_task()
                if task:
                    try:
                        self.dubber.process_video(task["path"])
                        self.db.update_status(task["id"], "DONE")
                    except Exception as e:
                        logger.exception(f"Worker failed task {task['id']}: {e}")
                        self.db.update_status(task["id"], "FAILED")
                else:
                    import time

                    time.sleep(5)

        def stop(self):
            self.stop_signal.set()

    worker = LegacyWorkerWrapper(dubber, db)
    worker.start()

    yield

    logger.info("Lifespan: Shutting down...")
    worker.stop()
    worker.join()


app = FastAPI(lifespan=lifespan)


@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    tasks = db.get_all_tasks()
    return templates.TemplateResponse("index.html", {"request": request, "tasks": tasks, "worker_alive": True})


@app.post("/retry/{task_id}")
async def retry_task(task_id: int):
    db.retry_task(task_id)
    return RedirectResponse(url="/", status_code=303)


@app.post("/delete/{task_id}")
async def delete_task(task_id: int):
    db.delete_task(task_id)
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
