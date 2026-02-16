import os
import logging
import threading
import sqlite3
import time
import uvicorn
from datetime import datetime
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, RedirectResponse
from pydantic import BaseModel
from typing import Optional

from main import AIDubber
from config import setup_logging

# Setup logging
setup_logging()
logger = logging.getLogger("Server")

# Database Configuration
DB_PATH = "/config/queue.db"
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

# Global control
stop_event = threading.Event()
worker_thread = None

# Templates
templates = Jinja2Templates(directory="templates")


def init_db():
    """Initializes the SQLite database for the task queue."""
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


class WebhookPayload(BaseModel):
    path: str
    eventType: Optional[str] = None


class DubberWorker(threading.Thread):
    def __init__(self, dubber: AIDubber):
        super().__init__()
        self.dubber = dubber
        self.daemon = True

    def run(self):
        logger.info("Worker: Started polling database for tasks.")
        while not stop_event.is_set():
            try:
                task = self.fetch_next_task()
                if not task:
                    time.sleep(5)  # Poll interval
                    continue

                task_id, path = task
                logger.info(f"Worker: Processing Task #{task_id}: {path}")

                try:
                    if os.path.exists(path):
                        self.dubber.process_video(path)
                        self.update_status(task_id, "DONE")
                        logger.info(f"Worker: Finished Task #{task_id}")
                    else:
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
            c.execute(
                "SELECT id, path FROM tasks WHERE status = 'QUEUED' ORDER BY created_at ASC LIMIT 1"
            )
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
    global worker_thread
    init_db()

    # Check for unfinished tasks (e.g. from crash) and reset them
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("UPDATE tasks SET status = 'QUEUED' WHERE status = 'PROCESSING'")
    if c.rowcount > 0:
        logger.warning(f"Reset {c.rowcount} interrupted tasks to QUEUED state.")
    conn.commit()
    conn.close()

    dubber = AIDubber()
    worker_thread = DubberWorker(dubber)
    worker_thread.start()

    yield

    # Shutdown
    stop_event.set()
    if worker_thread.is_alive():
        worker_thread.join()


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
            "worker_alive": worker_thread.is_alive() if worker_thread else False,
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
async def receive_webhook(payload: WebhookPayload):
    if not payload.path:
        raise HTTPException(status_code=400, detail="Path is required")

    logger.info(f"API: Received task for {payload.path}")

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    try:
        # Check if already exists (to avoid duplicates)
        c.execute("SELECT status FROM tasks WHERE path = ?", (payload.path,))
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
                    (datetime.now(), payload.path),
                )
                conn.commit()
                return {"status": "re-queued", "path": payload.path}
        else:
            c.execute("INSERT INTO tasks (path) VALUES (?)", (payload.path,))
            conn.commit()
            return {"status": "queued", "path": payload.path}
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
        "worker_alive": worker_thread.is_alive() if worker_thread else False,
    }


if __name__ == "__main__":
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host=host, port=port)
