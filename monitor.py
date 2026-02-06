import threading
import time
import subprocess
import logging

class ResourceMonitor(threading.Thread):
    def __init__(self, state_ref=None, interval=5.0):
        super().__init__()
        self.interval = interval
        self.state = state_ref or {}
        self.stop_event = threading.Event()

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
                        ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
                        capture_output=True, text=True
                    )
                    if res.returncode == 0:
                        lines = res.stdout.strip().split('\n')
                        for line in lines:
                            parts = line.split(',')
                            if len(parts) >= 2:
                                gpus.append((parts[0].strip(), parts[1].strip()))
                except:
                    pass
                
                # Fill missing GPUs with 0
                while len(gpus) < 2: gpus.append(("0", "0"))
                
                # Queue Sizes
                qt = self.state.get('q_text').qsize() if self.state.get('q_text') else 0
                qa = self.state.get('q_audio').qsize() if self.state.get('q_audio') else 0
                
                logging.info(f"[Monitor] CPU:{cpu}% RAM:{ram}% | GPU0:{gpus[0][0]}% Mem:{gpus[0][1]}MB | GPU1:{gpus[1][0]}% Mem:{gpus[1][1]}MB | Q_Text:{qt} Q_Audio:{qa}")
                    
            except Exception as e:
                logging.error(f"Monitor Error: {e}")
            
            time.sleep(self.interval)

    def stop(self):
        self.stop_event.set()