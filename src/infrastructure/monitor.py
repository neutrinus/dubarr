import threading
import time
import subprocess
import logging


class ResourceMonitor(threading.Thread):
    def __init__(self, state_ref=None, interval=10.0, llm_service=None, tts_service=None):
        super().__init__()
        self.interval = interval
        self.state = state_ref or {}
        self.stop_event = threading.Event()
        self.llm_service = llm_service
        self.tts_service = tts_service
        self.orchestrator_queue_size = 0

    def run(self):
        import psutil

        while not self.stop_event.is_set():
            try:
                cpu = psutil.cpu_percent(interval=None)
                ram = psutil.virtual_memory().percent

                # GPU Query via nvidia-smi
                gpus = []
                try:
                    res = subprocess.run(
                        [
                            "nvidia-smi",
                            "--query-gpu=utilization.gpu,memory.used,memory.free,memory.total",
                            "--format=csv,noheader,nounits",
                        ],
                        capture_output=True,
                        text=True,
                    )
                    if res.returncode == 0:
                        lines = res.stdout.strip().split("\n")
                        for line in lines:
                            parts = line.split(",")
                            if len(parts) >= 4:
                                gpus.append(
                                    {
                                        "util": parts[0].strip(),
                                        "used": parts[1].strip(),
                                        "free": parts[2].strip(),
                                        "total": parts[3].strip(),
                                    }
                                )
                except Exception:
                    pass

                # Fill missing GPUs with 0
                while len(gpus) < 2:
                    gpus.append({"util": "0", "used": "0", "free": "0", "total": "0"})

                # Queue sizes
                q_info = ""
                if self.llm_service or self.tts_service or self.orchestrator_queue_size > 0:
                    llm_q = self.llm_service.get_queue_size() if self.llm_service else 0
                    tts_q = self.tts_service.get_queue_size() if self.tts_service else 0
                    q_info = f" | QUEUE: LLM:{llm_q} TTS:{tts_q} PENDING:{self.orchestrator_queue_size}"

                logging.info(
                    f"[Monitor] CPU:{cpu}% RAM:{ram}% | "
                    f"GPU0:{gpus[0]['util']}% Mem:{gpus[0]['used']}/{gpus[0]['total']}MB (Free:{gpus[0]['free']}MB) | "
                    f"GPU1:{gpus[1]['util']}% Mem:{gpus[1]['used']}/{gpus[1]['total']}MB (Free:{gpus[1]['free']}MB)"
                    f"{q_info}"
                )

            except Exception as e:
                logging.error(f"Monitor Error: {e}")

            time.sleep(self.interval)

    def stop(self):
        self.stop_event.set()
