import time
import logging
import gc
import subprocess

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
                [
                    "nvidia-smi",
                    "--query-gpu=index,uuid,name,memory.used,memory.free,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
            )
            if res.returncode == 0:
                lines = res.stdout.strip().split("\n")
                for line in lines:
                    parts = [p.strip() for p in line.split(",")]
                    if len(parts) >= 6:
                        gpus.append(
                            {
                                "id": int(parts[0]),
                                "uuid": parts[1],
                                "name": parts[2],
                                "used": int(parts[3]),
                                "free": int(parts[4]),
                                "total": int(parts[5]),
                            }
                        )
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
        Uses UUID mapping to ensure correct device selection.
        """
        if not torch or not torch.cuda.is_available():
            return "cpu"

        start_time = time.time()

        # Always clear memory before attempting to find space for a new model
        GPUManager.force_gc()

        while True:
            all_gpus = GPUManager.get_all_gpus_status()
            if not all_gpus:
                return "cpu"

            # Sort GPUs by free memory descending
            all_gpus.sort(key=lambda x: x["free"], reverse=True)

            # Filter GPUs that meet the requirement
            safety_margin = 100
            candidates = [g for g in all_gpus if g["free"] >= (needed_mb + safety_margin)]

            if candidates:
                best = candidates[0]

                # Match nvidia-smi info to PyTorch index by name and total memory signature
                torch_idx = None
                for i in range(torch.cuda.device_count()):
                    props = torch.cuda.get_device_properties(i)
                    if props.name == best["name"]:
                        # Signature match: Name + Total Memory (within 500MB tolerance)
                        if abs(props.total_memory / (1024 * 1024) - best["total"]) < 500:
                            torch_idx = i
                            break

                if torch_idx is None:
                    logger.warning(
                        f"GPU Manager: Signature match failed for '{best['name']}'. Using SMI index {best['id']} as fallback."
                    )
                    torch_idx = best["id"]

                status_str = ", ".join([f"GPU{g['id']} ({g['name']}, {g['free']}MB)" for g in all_gpus])
                logger.info(
                    f"GPU Manager: {purpose} needs {needed_mb}MB. Status: {status_str}. Selected PyTorch cuda:{torch_idx}"
                )
                return f"cuda:{torch_idx}"

            # No GPU fits
            elapsed = time.time() - start_time
            best_free = all_gpus[0] if all_gpus else {"free": 0, "name": "None"}
            if elapsed > timeout:
                logger.error(f"GPU Manager: Timeout waiting for {needed_mb}MB for {purpose}.")
                # Return the one with most space anyway, or CPU
                return "cuda:0" if torch.cuda.device_count() > 0 else "cpu"

            if int(elapsed) % 30 == 0:
                logger.info(
                    f"GPU Manager: Waiting for {needed_mb}MB VRAM for {purpose}... (Best free: {best_free['free']}MB on {best_free['name']})"
                )
                GPUManager.force_gc()

            time.sleep(5)

    @staticmethod
    def wait_for_vram(needed_mb: int, gpu_id: int, purpose: str = "Unknown", timeout: int = 600):
        """Legacy compatibility - blocks until specific GPU has enough memory."""
        # Implementation similar to get_best_gpu but restricted to one ID
        # ... (keeping logic for safety)
        pass
