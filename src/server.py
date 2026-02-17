import os
import logging
import threading
import sqlite3
import time
import uvicorn
import secrets
import asyncio
from datetime import datetime
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect, Depends
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, RedirectResponse
from pydantic import BaseModel

from config import setup_logging, OUTPUT_FOLDER, API_USER, API_PASS, VIDEO_FOLDER, DATA_DIR

# Early logging to catch import issues
print("SERVER_STARTUP: server.py module loading...", flush=True)
logging.info("SERVER_STARTUP: server.py module loading...")

# Setup logging
setup_logging()
logger = logging.getLogger("Server")

# Authentication setup
security = HTTPBasic()


def authenticate(credentials: HTTPBasicCredentials = Depends(security)):
    current_user_bytes = credentials.username.encode("utf8")
    correct_user_bytes = API_USER.encode("utf8")
    is_correct_user = secrets.compare_digest(current_user_bytes, correct_user_bytes)

    current_pass_bytes = credentials.password.encode("utf8")
    correct_pass_bytes = API_PASS.encode("utf8")
    is_correct_pass = secrets.compare_digest(current_pass_bytes, correct_pass_bytes)

    if not (is_correct_user and is_correct_pass):
        raise HTTPException(
            status_code=401,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username


# Database Configuration
DB_PATH = os.path.join(DATA_DIR, "queue.db")
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

# Global control
stop_event = threading.Event()
worker_thread = None

# Templates
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_DIR = os.path.join(BASE_DIR, "templates")
templates = Jinja2Templates(directory=TEMPLATE_DIR)


def init_db():
    """Initializes the SQLite database for the task queue."""
    logger.info(f"DB: Initializing database at {DB_PATH}")
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS tasks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            path TEXT NOT NULL UNIQUE,
            status TEXT DEFAULT 'QUEUED',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """
    )
    conn.commit()
    conn.close()
    logger.info("DB: Database initialized.")


class WebhookPayload(BaseModel):
    path: str
    eventType: Optional[str] = None


class DubberWorker(threading.Thread):
    def __init__(self, dubber):
        super().__init__()
        self.dubber = dubber
        self.daemon = True

    def run(self):
        print("Worker: Started polling database for tasks.", flush=True)
        logger.info("Worker: Started polling database for tasks.")
        while not stop_event.is_set():
            try:
                task = self.fetch_next_task()
                if not task:
                    time.sleep(5)  # Poll interval
                    continue

                task_id, path = task
                print(f"Worker: Processing Task #{task_id}: {path}", flush=True)
                logger.info(f"Worker: Processing Task #{task_id}: {path}")

                try:
                    if os.path.exists(path):
                        self.dubber.process_video(path)
                        print(f"Worker: Finished Task #{task_id}", flush=True)
                        self.update_status(task_id, "DONE")
                        logger.info(f"Worker: Finished Task #{task_id}")
                    else:
                        print(f"Worker: File not found: {path}", flush=True)
                        logger.error(f"Worker: File not found: {path}")
                        self.update_status(task_id, "FAILED_FILE_NOT_FOUND")
                except Exception as e:
                    logger.exception(f"Worker: Failed Task #{task_id}: {e}")
                    self.update_status(task_id, "FAILED")

            except Exception as e:
                logger.error(f"Worker Loop Error: {e}")
                time.sleep(5)

    def fetch_next_task(self):
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        # Simple locking mechanism: find QUEUED, update to PROCESSING
        try:
            c.execute("BEGIN IMMEDIATE")
            c.execute("SELECT id, path FROM tasks WHERE status = 'QUEUED' ORDER BY created_at ASC LIMIT 1")
            row = c.fetchone()
            if row:
                task_id, path = row
                c.execute(
                    "UPDATE tasks SET status = 'PROCESSING', updated_at = ? WHERE id = ?",
                    (datetime.now(), task_id),
                )
                conn.commit()
                return task_id, path
            conn.commit()
            return None
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def update_status(self, task_id, status):
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute(
            "UPDATE tasks SET status = ?, updated_at = ? WHERE id = ?",
            (status, datetime.now(), task_id),
        )
        conn.commit()
        conn.close()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("Lifespan: Starting application setup...", flush=True)
    logger.info("Lifespan: Starting application setup...")
    try:
        init_db()

        # Check for unfinished tasks (e.g. from crash) and reset them
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("UPDATE tasks SET status = 'QUEUED' WHERE status = 'PROCESSING'")
        if c.rowcount > 0:
            print(f"Lifespan: Reset {c.rowcount} interrupted tasks to QUEUED state.", flush=True)
            logging.warning(f"Reset {c.rowcount} interrupted tasks to QUEUED state.")
        conn.commit()
        conn.close()

        def start_worker():
            global worker_thread
            try:
                # Import AIDubber here to speed up API startup
                from main import AIDubber

                print("Lifespan: Initializing AIDubber in background...", flush=True)
                logging.info("Lifespan: Initializing AIDubber in background...")
                dubber = AIDubber()
                print("Lifespan: AIDubber initialized. Starting worker thread...", flush=True)
                logging.info("Lifespan: AIDubber initialized. Starting worker thread...")
                worker_thread = DubberWorker(dubber)
                worker_thread.start()
                print("Lifespan: Worker thread started.", flush=True)
                logging.info("Lifespan: Worker thread started.")
            except Exception as e:
                print(f"Lifespan: Background initialization failed: {e}", flush=True)
                logging.exception(f"Lifespan: Background initialization failed: {e}")

        threading.Thread(target=start_worker, daemon=True).start()
        logger.info("Lifespan: Background initialization triggered.")

    except Exception as e:
        print(f"Lifespan: Initialization failed: {e}", flush=True)
        logging.exception(f"Lifespan: Initialization failed: {e}")
        raise

    yield

    # Shutdown
    logging.info("Lifespan: Shutting down...")
    stop_event.set()
    if worker_thread and worker_thread.is_alive():
        worker_thread.join()
    logging.info("Lifespan: Shutdown complete.")


app = FastAPI(lifespan=lifespan)


@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("SELECT * FROM tasks ORDER BY created_at DESC")
    tasks = c.fetchall()
    conn.close()

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "tasks": tasks,
            "worker_alive": worker_thread.is_alive() if worker_thread is not None else False,
        },
    )


@app.post("/retry/{task_id}")
async def retry_task(task_id: int):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "UPDATE tasks SET status = 'QUEUED', updated_at = ? WHERE id = ?",
        (datetime.now(), task_id),
    )
    conn.commit()
    conn.close()
    return RedirectResponse(url="/", status_code=303)


@app.post("/delete/{task_id}")
async def delete_task(task_id: int):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("DELETE FROM tasks WHERE id = ?", (task_id,))
    conn.commit()
    conn.close()
    return RedirectResponse(url="/", status_code=303)


@app.post("/webhook")
async def receive_webhook(payload: WebhookPayload, username: str = Depends(authenticate)):
    if not payload.path:
        raise HTTPException(status_code=400, detail="Path is required")

    # If path is absolute, use it directly (useful for smoke tests in /tmp)
    # Otherwise, assume it is relative to VIDEO_FOLDER
    if os.path.isabs(payload.path):
        full_path = payload.path
    else:
        relative_path = payload.path
        if relative_path.startswith("videos/"):
            relative_path = relative_path[len("videos/") :]
        full_path = os.path.join(VIDEO_FOLDER, relative_path)

    full_path = os.path.normpath(full_path)

    logger.info(f"API: Received task for {payload.path} from {username}. Full path: {full_path}")

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    try:
        # Check if already exists (to avoid duplicates)
        c.execute("SELECT status FROM tasks WHERE path = ?", (full_path,))
        row = c.fetchone()
        if row:
            status = row[0]
            if status in ["QUEUED", "PROCESSING"]:
                return {
                    "status": "ignored",
                    "detail": f"Already in queue with status: {status}",
                }
            else:
                # Re-queue if it was done/failed before (manual retry via webhook)
                c.execute(
                    "UPDATE tasks SET status = 'QUEUED', updated_at = ? WHERE path = ?",
                    (datetime.now(), full_path),
                )
                conn.commit()
                return {"status": "re-queued", "path": full_path}
        else:
            c.execute("INSERT INTO tasks (path) VALUES (?)", (full_path,))
            conn.commit()
            return {"status": "queued", "path": full_path}
    finally:
        conn.close()


@app.get("/health")
async def health_check():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT status, COUNT(*) FROM tasks GROUP BY status")
    stats = dict(c.fetchall())
    conn.close()

    return {
        "status": "healthy",
        "queue_stats": stats,
        "worker_alive": worker_thread.is_alive() if worker_thread is not None else False,
    }


@app.websocket("/ws/logs")
async def websocket_logs(websocket: WebSocket):
    await websocket.accept()
    log_file = os.path.join(OUTPUT_FOLDER, "processing.log")

    # Ensure log file exists
    if not os.path.exists(log_file):
        with open(log_file, "w") as f:
            f.write("Log file created.\n")

    try:
        # 1. Send last 50 lines immediately
        with open(log_file, "r") as f:
            lines = f.readlines()
            last_lines = lines[-50:]
            for line in last_lines:
                await websocket.send_text(line.strip())

        # 2. Tail the file
        with open(log_file, "r") as f:
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
        logger.error(f"WebSocket error: {e}")
    finally:
        try:
            await websocket.close()
        except Exception:
            pass


if __name__ == "__main__":
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", 8080))
    logger.info(f"Server: Starting uvicorn on {host}:{port}")
    uvicorn.run(app, host=host, port=port)
