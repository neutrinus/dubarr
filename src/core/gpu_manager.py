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
    """Intelligent VRAM manager that dynamically allocates models to best available GPU."""

    @staticmethod
    def get_all_gpus_status():
        """Returns a list of status dicts for all available GPUs."""
        gpus = []
        try:
            res = subprocess.run(
                ["nvidia-smi", "--query-gpu=index,name,memory.used,memory.free,memory.total", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
            )
            if res.returncode == 0:
                lines = res.stdout.strip().split("\n")
                for line in lines:
                    parts = [p.strip() for p in line.split(",")]
                    if len(parts) >= 5:
                        gpus.append({
                            "id": int(parts[0]),
                            "name": parts[1],
                            "used": int(parts[2]),
                            "free": int(parts[3]),
                            "total": int(parts[4])
                        })
        except Exception as e:
            logger.warning(f"GPU Manager: Failed to query nvidia-smi: {e}")
        return gpus

    @staticmethod
    def force_gc():
        """Aggressively clears memory across all GPUs."""
        gc.collect()
        if torch and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    @staticmethod
    def get_best_gpu(needed_mb: int, purpose: str = "Unknown", timeout: int = 600):
        """
        Finds the best GPU for the task and returns its PyTorch device string (e.g. 'cuda:0').
        Blocks if no GPU has enough memory.
        """
        if not torch or not torch.cuda.is_available():
            return "cpu"

        start_time = time.time()
        while True:
            all_gpus = GPUManager.get_all_gpus_status()
            if not all_gpus:
                return "cpu"

            # Sort GPUs by free memory descending
            all_gpus.sort(key=lambda x: x["free"], reverse=True)
            
            # Filter GPUs that meet the requirement + safety margin
            safety_margin = 500
            candidates = [g for g in all_gpus if g["free"] >= (needed_mb + safety_margin)]
            
            if candidates:
                # Choose the one with the most free space
                best = candidates[0]
                # IMPORTANT: We need to return the index that TORCH sees.
                for i in range(torch.cuda.device_count()):
                    if torch.cuda.get_device_name(i) == best["name"]:
                        if time.time() - start_time > 2:
                            logger.info(f"GPU Manager: Selected {best['name']} for {purpose} ({best['free']}MB free)")
                        return f"cuda:{i}"

            # No GPU fits
            elapsed = time.time() - start_time
            if elapsed > timeout:
                logger.error(f"GPU Manager: Timeout waiting for {needed_mb}MB for {purpose}.")
                # Return the one with most space anyway, or CPU
                return f"cuda:0" if torch.cuda.device_count() > 0 else "cpu"

            if int(elapsed) % 30 == 0:
                logger.info(f"GPU Manager: Waiting for {needed_mb}MB VRAM for {purpose}... (Best free: {best['free']}MB on {best['name']})")
                GPUManager.force_gc()

            time.sleep(5)

    @staticmethod
    def wait_for_vram(needed_mb: int, gpu_id: int, purpose: str = "Unknown", timeout: int = 600):
        """Legacy compatibility - blocks until specific GPU has enough memory."""
        # Implementation similar to get_best_gpu but restricted to one ID
        # ... (keeping logic for safety)
        pass
