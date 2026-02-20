import threading
import queue
import logging
import concurrent.futures
import itertools
from typing import Any, Callable, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

class GPUService(threading.Thread):
    """
    A base class for GPU-bound services that handle tasks via a priority queue.
    Ensures serialized access to a specific GPU resource while allowing multiple
    clients to submit tasks asynchronously.
    """
    def __init__(self, name: str):
        super().__init__(name=f"Service-{name}", daemon=True)
        self.service_name = name
        self.queue = queue.PriorityQueue()
        self._stop_event = threading.Event()
        self._counter = itertools.count()  # Tie-breaker for PriorityQueue

    def submit(self, func: Callable, *args, priority: int = 10, **kwargs) -> concurrent.futures.Future:
        """
        Submits a task to the service.
        Returns a Future object that will contain the result.
        Lower priority number means higher priority (standard PriorityQueue behavior).
        """
        if self._stop_event.is_set():
            raise RuntimeError(f"Service {self.service_name} is stopped.")

        future = concurrent.futures.Future()
        # Entry format: (priority, tie_breaker, (func, args, kwargs, future))
        entry = (priority, next(self._counter), (func, args, kwargs, future))
        self.queue.put(entry)
        return future

    def stop(self):
        """Signals the service thread to stop."""
        self._stop_event.set()
        # Put a sentinel with highest priority to wake up and exit
        self.queue.put((-1, next(self._counter), None))

    def run(self):
        logger.info(f"GPU Service [{self.service_name}] started.")
        while not self._stop_event.is_set():
            try:
                # Use a timeout so we can check the stop_event periodically
                priority, _, task = self.queue.get(timeout=1.0)
                
                if task is None:
                    break
                
                func, args, kwargs, future = task
                
                if future.set_running_or_notify_cancel():
                    try:
                        # logger.debug(f"Service [{self.service_name}] executing task with priority {priority}")
                        result = func(*args, **kwargs)
                        future.set_result(result)
                    except Exception as e:
                        logger.error(f"Service [{self.service_name}] task failed: {e}")
                        future.set_exception(e)
                
                self.queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Service [{self.service_name}] loop error: {e}")

        logger.info(f"GPU Service [{self.service_name}] stopped.")

class LLMService(GPUService):
    """Dedicated service for LLM inference (Gemma/Llama)."""
    def __init__(self):
        super().__init__("LLM")

class TTSService(GPUService):
    """Dedicated service for TTS synthesis (XTTS)."""
    def __init__(self):
        super().__init__("TTS")
