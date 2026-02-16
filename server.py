import os
import logging
import threading
import queue
import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional

from main import AIDubber
from config import setup_logging

# Setup logging properly for the server
setup_logging()
logger = logging.getLogger("Server")

# Queue for video processing tasks
# Items in queue: {"path": str, "source_lang": str (opt), "imdb_id": str (opt)}
task_queue = queue.Queue()

# Global worker thread
worker_thread = None
stop_event = threading.Event()


class WebhookPayload(BaseModel):
    path: str
    eventType: Optional[str] = None  # Sonarr/Radarr event type (Grab, Download, etc.)

    # Radarr/Sonarr specific fields might be needed depending on what they send
    # For now, we mainly need the path.
    # Example Radarr payload for "Download":
    # { "movie": { "title": "...", "folderPath": "..." }, "remoteMovie": { ... }, "eventType": "Download" }
    # But usually custom scripts pass arguments via env vars or simple JSON if configured so.
    # We will assume a simplified payload for manual testing or a specific connect setup:
    # { "path": "/movies/Movie/file.mkv" }


class DubberWorker(threading.Thread):
    def __init__(self, dubber: AIDubber):
        super().__init__()
        self.dubber = dubber
        self.daemon = True

    def run(self):
        logger.info("Worker: Started listening for tasks.")
        while not stop_event.is_set():
            try:
                task = task_queue.get(timeout=2)
            except queue.Empty:
                continue

            path = task.get("path")
            if not path:
                task_queue.task_done()
                continue

            logger.info(f"Worker: Processing {path}")
            try:
                if os.path.exists(path):
                    self.dubber.process_video(path)
                    logger.info(f"Worker: Finished {path}")
                else:
                    logger.error(f"Worker: File not found: {path}")
            except Exception as e:
                logger.exception(f"Worker: Failed to process {path}: {e}")
            finally:
                task_queue.task_done()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global worker_thread
    dubber = AIDubber()
    worker_thread = DubberWorker(dubber)
    worker_thread.start()
    yield
    # Shutdown
    stop_event.set()
    if worker_thread.is_alive():
        worker_thread.join()


app = FastAPI(lifespan=lifespan)


@app.post("/webhook")
async def receive_webhook(payload: WebhookPayload):
    """
    Receives a webhook from *arr or manual trigger.
    Expected JSON: {"path": "/path/to/movie.mkv"}
    """
    if not payload.path:
        raise HTTPException(status_code=400, detail="Path is required")

    logger.info(f"API: Received task for {payload.path}")
    task_queue.put({"path": payload.path})
    return {"status": "queued", "path": payload.path}


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "queue_size": task_queue.qsize(),
        "worker_alive": worker_thread.is_alive() if worker_thread else False,
    }


if __name__ == "__main__":
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host=host, port=port)
