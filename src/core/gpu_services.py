import threading
import queue
import logging
import concurrent.futures
import itertools
from typing import Callable

logger = logging.getLogger(__name__)


class GPUService(threading.Thread):
    """
    A base class for GPU-bound services that handle tasks via a priority queue.
    Ensures managed access to a specific GPU resource.
    """

    def __init__(self, name: str, num_workers: int = 1):
        super().__init__(name=f"Service-{name}-Manager", daemon=True)
        self.service_name = name
        self.queue = queue.PriorityQueue()
        self._stop_event = threading.Event()
        self._counter = itertools.count()
        self.num_workers = num_workers
        self._worker_threads = []

    def submit(self, func: Callable, *args, priority: int = 10, **kwargs) -> concurrent.futures.Future:
        if self._stop_event.is_set():
            raise RuntimeError(f"Service {self.service_name} is stopped.")

        future = concurrent.futures.Future()
        entry = (priority, next(self._counter), (func, args, kwargs, future))
        self.queue.put(entry)
        return future

    def start(self):
        """Starts the manager and worker threads."""
        for i in range(self.num_workers):
            t = threading.Thread(target=self._worker_loop, name=f"Worker-{self.service_name}-{i}", daemon=True)
            t.start()
            self._worker_threads.append(t)
        super().start()

    def stop(self):
        self._stop_event.set()
        for _ in range(self.num_workers + 1):
            self.queue.put((-1, next(self._counter), None))

    def _worker_loop(self):
        while not self._stop_event.is_set():
            try:
                priority, _, task = self.queue.get(timeout=1.0)
                if task is None:
                    break

                func, args, kwargs, future = task
                if future.set_running_or_notify_cancel():
                    try:
                        result = func(*args, **kwargs)
                        future.set_result(result)
                    except Exception as e:
                        logger.error(f"Service [{self.service_name}] task failed: {e}")
                        future.set_exception(e)
                self.queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Service [{self.service_name}] worker error: {e}")

    def run(self):
        """Manager thread just waits for stop signal."""
        logger.info(f"GPU Service [{self.service_name}] started with {self.num_workers} workers.")
        self._stop_event.wait()
        logger.info(f"GPU Service [{self.service_name}] stopping...")


class LLMService(GPUService):
    """Dedicated service for LLM inference. Supports parallel slots."""

    def __init__(self, num_workers: int = 2):
        super().__init__("LLM", num_workers=num_workers)


class TTSService(GPUService):
    """Dedicated service for TTS synthesis. Sequential preferred for XTTS stability."""

    def __init__(self):
        super().__init__("TTS", num_workers=1)
