import time
import logging
import gc
import subprocess
import threading

try:
    import torch
except ImportError:
    torch = None

logger = logging.getLogger(__name__)

class GPUManager:
    """Helper class to manage VRAM allocation and prevent OOM errors."""

    @staticmethod
    def get_gpu_status(gpu_id: int):
        """Returns {used, free, total} in MB for the specified GPU, matching by name if needed."""
        try:
            # We match by name to be 100% sure because torch and nvidia-smi indices often differ
            import torch
            target_name = torch.cuda.get_device_name(gpu_id)
            
            res = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.used,memory.free,memory.total", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
            )
            if res.returncode == 0:
                lines = res.stdout.strip().split("
")
                for line in lines:
                    parts = [p.strip() for p in line.split(",")]
                    if len(parts) >= 4 and parts[0] == target_name:
                        return {
                            "used": int(parts[1]),
                            "free": int(parts[2]),
                            "total": int(parts[3])
                        }
        except Exception as e:
            logger.warning(f"GPU Manager: Failed to query nvidia-smi for {gpu_id}: {e}")
        return None

    @staticmethod
    def force_gc():
        """Aggressively clears memory."""
        gc.collect()
        if torch and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    @staticmethod
    def wait_for_vram(needed_mb: int, gpu_id: int, purpose: str = "Unknown", timeout: int = 600, check_interval: int = 5):
        """Blocks execution until sufficient VRAM is available."""
        if gpu_id is None or gpu_id < 0:
            return # CPU mode or invalid ID

        start_time = time.time()
        
        while True:
            status = GPUManager.get_gpu_status(gpu_id)
            if not status:
                logger.warning(f"GPU Manager: Could not get status for GPU {gpu_id} ({purpose}). Proceeding blindly.")
                return

            if status["free"] >= needed_mb:
                if time.time() - start_time > 5:
                     logger.info(f"GPU Manager: VRAM available for {purpose}! ({status['free']}MB free >= {needed_mb}MB needed)")
                return

            elapsed = time.time() - start_time
            if elapsed > timeout:
                logger.error(f"GPU Manager: Timeout waiting for VRAM for {purpose} on GPU {gpu_id}. Free: {status['free']}MB, Needed: {needed_mb}MB.")
                # Try to force GC one last time and proceed, hoping for the best
                GPUManager.force_gc()
                return

            if int(elapsed) % 30 == 0:
                logger.info(f"GPU Manager: Waiting for {needed_mb}MB VRAM for {purpose} on GPU {gpu_id}... (Current Free: {status['free']}MB)")
                GPUManager.force_gc()

            time.sleep(check_interval)
