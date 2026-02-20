import os
import logging
import sys

try:
    import torch
except ImportError:
    torch = None

# --- CONFIGURATION & HARDWARE DETECTION ---

DEBUG_MODE = os.environ.get("DEBUG", "0").lower() in ("1", "true", "yes")
VERBOSE_MODE = os.environ.get("VERBOSE", "0").lower() in ("1", "true", "yes")
MOCK_MODE = os.environ.get("MOCK_MODE", "0").lower() in ("1", "true", "yes")
LOG_LEVEL = logging.DEBUG if VERBOSE_MODE else logging.INFO

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.environ.get("DATA_DIR", "/app/data")

VIDEO_FOLDER = os.path.join(BASE_DIR, "videos")
OUTPUT_FOLDER = os.path.join(DATA_DIR, "logs")
DB_PATH = os.path.join(DATA_DIR, "queue.db")
TEMP_DIR = os.path.join(DATA_DIR, "temp")
MODEL_PATH = os.path.join(DATA_DIR, "models", "google_gemma-3-12b-it-Q4_K_M.gguf")

print(f"DEBUG: config.BASE_DIR={BASE_DIR}", flush=True)
print(f"DEBUG: config.DATA_DIR={DATA_DIR}", flush=True)
print(f"DEBUG: config.VIDEO_FOLDER={VIDEO_FOLDER}", flush=True)
print(f"DEBUG: config.OUTPUT_FOLDER={OUTPUT_FOLDER}", flush=True)
print(f"DEBUG: config.MODEL_PATH={MODEL_PATH}", flush=True)

WHISPER_MODEL = "large-v3-turbo"


# Safe logging setup
def setup_logging():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    os.makedirs(VIDEO_FOLDER, exist_ok=True)
    os.makedirs(TEMP_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    logging.basicConfig(
        level=LOG_LEVEL,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.path.join(OUTPUT_FOLDER, "processing.log"), mode="a", encoding="utf-8"),
        ],
        force=True,
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

# Fail early if HF_TOKEN is missing (unless in MOCK_MODE)
if not HF_TOKEN and not MOCK_MODE:
    print("\n" + "!" * 60)
    print("FATAL ERROR: HF_TOKEN environment variable is missing!")
    print("Pyannote Diarization requires a Hugging Face token.")
    print("Please set HF_TOKEN in your environment or docker-compose.yml.")
    print("!" * 60 + "\n")
    sys.exit(1)

# API Authentication
API_USER = os.environ.get("API_USER", "dubarr")
API_PASS = os.environ.get("API_PASS", "dubarr")

# --- HARDWARE DETECTION ---


def get_compute_device_type():
    if torch and torch.cuda.is_available():
        return "cuda"
    return "cpu"


DEVICE_TYPE = get_compute_device_type()
# If we have 1 GPU or CPU, we must use a lock to prevent concurrent inference on the same device.
# For multi-GPU, GPUManager will handle isolation if needed, but we still use lock for safety within a single process.
USE_LOCK = DEVICE_TYPE == "cpu" or (torch and torch.cuda.device_count() == 1)
STRATEGY = "dynamic"  # Handled by GPUManager
