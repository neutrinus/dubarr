import logging
import threading
import time
from typing import Optional
from infrastructure.database import Database

logger = logging.getLogger(__name__)


class JobWorker:
    def __init__(self, db: Database, pipeline_factory):
        """
        worker logic that polls DB and runs the pipeline.
        pipeline_factory: a callable that returns a DubbingPipeline instance
        """
        self.db = db
        self.pipeline_factory = pipeline_factory
        self.stop_event = threading.Event()
        self.thread: Optional[threading.Thread] = None

    def start(self):
        if self.thread and self.thread.is_alive():
            return
        self.stop_event.clear()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        logger.info("JobWorker: Started.")

    def stop(self):
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=10)
        logger.info("JobWorker: Stopped.")

    def _run(self):
        while not self.stop_event.is_set():
            try:
                task = self.db.fetch_next_task()
                if not task:
                    time.sleep(5)
                    continue

                task_id = task["id"]
                task_path = task["path"]
                logger.info(f"Worker: Processing Task #{task_id}: {task_path}")

                try:
                    # Create a fresh pipeline for each task (or use injected one)
                    pipeline = self.pipeline_factory()
                    # Pass task_id to pipeline for checkpointing (later)
                    pipeline.process_video(task_path, task_id=task_id)
                    self.db.update_status(task_id, "DONE")
                    logger.info(f"Worker: Finished Task #{task_id}")
                except Exception as e:
                    logger.exception(f"Worker: Failed Task #{task_id}: {e}")
                    self.db.update_status(task_id, "FAILED")

            except Exception as e:
                logger.error(f"Worker Loop Error: {e}")
                time.sleep(5)
