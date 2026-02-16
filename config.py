import os
import logging
import torch

# GPU STRATEGY
# GPU 1: RTX 3070 (8GB)  -> DEDICATED LLM ENGINE (Pure Gemma 3 12B)
# GPU 0: RTX 3060 (12GB) -> MULTI-TASK ENGINE (Demucs, Whisper, Diarization, XTTS)

DEBUG_MODE = os.environ.get("DEBUG", "0").lower() in ("1", "true", "yes")
VERBOSE_MODE = os.environ.get("VERBOSE", "0").lower() in ("1", "true", "yes")
LOG_LEVEL = logging.DEBUG if VERBOSE_MODE else logging.INFO

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# For manual testing/watchfolder (legacy)
VIDEO_FOLDER = os.path.join(BASE_DIR, "videos")
# Output folder is now mainly for logs/debug artifacts in the new architecture
OUTPUT_FOLDER = os.environ.get("LOGS_DIR", os.path.join(BASE_DIR, "logs"))
TEMP_DIR = "/tmp/dubber"
MODEL_PATH = os.path.join(BASE_DIR, "models", "gemma-3-12b-it-Q4_K_M.gguf")
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

# Hardware Setup
GPU_COUNT = torch.cuda.device_count()
GPU_LLM = 1 if GPU_COUNT > 1 else 0
GPU_AUDIO = 0

TARGET_LANGS = os.environ.get("TARGET_LANGS", "pl").split(",")
HF_TOKEN = os.environ.get("HF_TOKEN")
