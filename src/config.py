import os
import logging
import torch

# --- CONFIGURATION & HARDWARE DETECTION ---

DEBUG_MODE = os.environ.get("DEBUG", "0").lower() in ("1", "true", "yes")
VERBOSE_MODE = os.environ.get("VERBOSE", "0").lower() in ("1", "true", "yes")
MOCK_MODE = os.environ.get("MOCK_MODE", "0").lower() in ("1", "true", "yes")
LOG_LEVEL = logging.DEBUG if VERBOSE_MODE else logging.INFO

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VIDEO_FOLDER = os.path.join(BASE_DIR, "videos")
OUTPUT_FOLDER = os.environ.get("LOGS_DIR", os.path.join(BASE_DIR, "logs"))
TEMP_DIR = "/tmp/dubber"
MODEL_PATH = os.path.join(BASE_DIR, "models", "google_gemma-3-12b-it-Q4_K_M.gguf")

print(f"DEBUG: config.BASE_DIR={BASE_DIR}")
print(f"DEBUG: config.VIDEO_FOLDER={VIDEO_FOLDER}")
print(f"DEBUG: config.OUTPUT_FOLDER={OUTPUT_FOLDER}")
print(f"DEBUG: config.MODEL_PATH={MODEL_PATH}")

WHISPER_MODEL = "large-v3"


# Safe logging setup
def setup_logging():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    logging.basicConfig(
        level=LOG_LEVEL,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.path.join(OUTPUT_FOLDER, "processing.log"), mode="a", encoding="utf-8"),
        ],
    )


os.environ["HF_HUB_READ_TIMEOUT"] = "120"
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "120"

LANG_MAP = {
    "pl": "Polish",
    "en": "English",
    "de": "German",
    "es": "Spanish",
    "fr": "French",
    "it": "Italian",
    "ja": "Japanese",
    "zh": "Chinese",
    "ru": "Russian",
}

TARGET_LANGS = os.environ.get("TARGET_LANGS", "pl").split(",")
HF_TOKEN = os.environ.get("HF_TOKEN")

# API Authentication
API_USER = os.environ.get("API_USER", "dubarr")
API_PASS = os.environ.get("API_PASS", "dubarr")

# --- DYNAMIC HARDWARE ALLOCATION ---


def get_compute_device():
    """Detects available hardware and assigns roles."""
    strategy = "sequential"  # Default safe strategy

    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()

        # Calculate Total VRAM across all GPUs being used
        total_vram_gb = 0
        for i in range(gpu_count):
            props = torch.cuda.get_device_properties(i)
            total_vram_gb += props.total_memory / (1024**3)

        logging.info(f"Hardware: Detected {gpu_count} GPU(s) with Total VRAM: {total_vram_gb:.2f} GB")

        # Threshold for parallel execution (LLM ~10GB + TTS ~4GB + Whisper ~3GB + Overhead)
        # We set it conservatively at 18GB
        if total_vram_gb > 18.0:
            strategy = "parallel"
            logging.info("Strategy: PARALLEL (Sufficient VRAM detected)")
        else:
            logging.info("Strategy: SEQUENTIAL (Limited VRAM, enabling safety locks)")

        if gpu_count >= 2:
            # Optimal: Dual GPU Split
            llm_idx = int(os.environ.get("GPU_LLM_ID", "1"))
            audio_idx = int(os.environ.get("GPU_AUDIO_ID", "0"))

            # Ensure indices exist
            if llm_idx >= gpu_count:
                llm_idx = 0
            if audio_idx >= gpu_count:
                audio_idx = 0

            return {
                "llm": f"cuda:{llm_idx}",
                "audio": f"cuda:{audio_idx}",
                "use_lock": False,
                "type": "cuda",
                "strategy": strategy,
            }
        else:
            # Single GPU Shared
            return {
                "llm": "cuda:0",
                "audio": "cuda:0",
                "use_lock": True,  # Critical to prevent compute clash
                "type": "cuda",
                "strategy": strategy,
            }
    else:
        # CPU Fallback
        logging.warning("Hardware: No GPU detected. Running in CPU mode (Slow!).")
        logging.info("Strategy: SEQUENTIAL (CPU Mode)")
        return {
            "llm": "cpu",
            "audio": "cpu",
            "use_lock": True,
            "type": "cpu",
            "strategy": "sequential",  # CPU must be sequential
        }


# Initialize hardware config
HW_CONFIG = get_compute_device()
DEVICE_LLM = HW_CONFIG["llm"]
DEVICE_AUDIO = HW_CONFIG["audio"]
USE_LOCK = HW_CONFIG["use_lock"]
DEVICE_TYPE = HW_CONFIG["type"]
STRATEGY = HW_CONFIG["strategy"]
