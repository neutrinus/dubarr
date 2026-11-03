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
LLM_MODEL_REPO = os.environ.get("LLM_MODEL_REPO", "bartowski/google_gemma-3-12b-it-GGUF")
LLM_MODEL_FILE = os.environ.get("LLM_MODEL_FILE", "google_gemma-3-12b-it-Q4_K_M.gguf")
MODEL_PATH = os.path.join(DATA_DIR, "models", LLM_MODEL_FILE)
# Optional external OpenAI-compatible server (e.g. llama.cpp's llama-server). When set, no model is loaded in-process.
LLM_API_URL = os.environ.get("LLM_API_URL", "")
LLM_API_MODEL = os.environ.get("LLM_API_MODEL", "")
LLM_API_KEY = os.environ.get("LLM_API_KEY", "")
LLM_TIMEOUT = int(os.environ.get("LLM_TIMEOUT", "600"))
LLM_N_CTX = int(os.environ.get("LLM_N_CTX", "8192"))
# Per-request completion budget for the bulk calls (analysis, ASR editor, drafts). A truncated answer
# triggers one retry with double the budget, so a reasoning model cannot silently swallow the whole reply.
LLM_MAX_TOKENS = int(os.environ.get("LLM_MAX_TOKENS", "4096"))
# Optional hint about the production, overriding the series/episode parsed from the file name.
SERIES_CONTEXT = os.environ.get("SERIES_CONTEXT", "")

print(f"DEBUG: config.BASE_DIR={BASE_DIR}", flush=True)
print(f"DEBUG: config.DATA_DIR={DATA_DIR}", flush=True)
print(f"DEBUG: config.VIDEO_FOLDER={VIDEO_FOLDER}", flush=True)
print(f"DEBUG: config.OUTPUT_FOLDER={OUTPUT_FOLDER}", flush=True)
print(f"DEBUG: config.MODEL_PATH={MODEL_PATH}", flush=True)

WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "large-v3-turbo")
# Second transcription pass without VAD over the gaps the first pass left (commentary under arena noise, which VAD
# drops); only confident, non-hallucinated segments are kept (Spec 001, 2.2).
# What Whisper transcribes: the original mix (default) or the Demucs vocals; diarization always uses the vocals
ASR_SOURCE = os.environ.get("ASR_SOURCE", "original").lower()
ASR_GAP_PASS = os.environ.get("ASR_GAP_PASS", "on").lower() not in ("0", "off", "false", "no")
ASR_GAP_MIN = float(os.environ.get("ASR_GAP_MIN", "2.0"))

# Seconds per TTS request (raise it on CPU, synthesis is slow there)
TTS_TIMEOUT = int(os.environ.get("TTS_TIMEOUT", "60"))

# Speech synthesis is OmniVoice. Its server runs as a separate process, on the app's own interpreter by default.
if os.environ.get("TTS_BACKEND", "omnivoice").lower() != "omnivoice":
    print("WARNING: TTS_BACKEND is no longer used, XTTS was removed; OmniVoice is the only TTS backend.", flush=True)
OMNIVOICE_PYTHON = os.environ.get("OMNIVOICE_PYTHON", sys.executable)

# Media metadata (show/episode or movie, synopsis, cast) as context for the translator.
# "on": .nfo files next to the video, then TVmaze (episodes, no account) / TMDB (movies, needs TMDB_API_KEY);
# "local": .nfo files only; "off": no lookup. Online lookups are skipped in MOCK_MODE.
METADATA_LOOKUP = os.environ.get("METADATA_LOOKUP", "on").lower()
if METADATA_LOOKUP in ("1", "true", "yes"):
    METADATA_LOOKUP = "on"
elif METADATA_LOOKUP in ("0", "false", "no"):
    METADATA_LOOKUP = "off"
TMDB_API_KEY = os.environ.get("TMDB_API_KEY", "")
METADATA_TIMEOUT = int(os.environ.get("METADATA_TIMEOUT", "10"))
# Where speech synthesis runs: "auto" picks the GPU with the most free VRAM, "cpu" keeps the GPUs free for
# something else (e.g. a shared LLM) and is much slower.
TTS_DEVICE = os.environ.get("TTS_DEVICE", "auto").lower()
TTS_CPU_THREADS = int(os.environ.get("TTS_CPU_THREADS", "0")) or (os.cpu_count() or 1)
# Long lines are synthesized in chunks at sentence boundaries. Short chunks keep the dub in sync and cap
TTS_MAX_CHUNK_SECONDS = float(os.environ.get("TTS_MAX_CHUNK_SECONDS", "10.0"))
# Reference-clip transcript sent to OmniVoice: "off" (default) keeps it out of cross-lingual cloning, where an
TTS_REF_TEXT = os.environ.get("TTS_REF_TEXT", "off").lower()


# Dubbed speech may run into the silence after a segment and, when still too long, start early in the one before it.
# Cap on how much trailing silence a segment may borrow, and the gap kept before the next speech.
SLACK_AFTER_MAX = float(os.environ.get("SLACK_AFTER_MAX", "1.5"))
# A take that is still too long may also start this much earlier, in the silence before its line.
SLACK_BEFORE_MAX = float(os.environ.get("SLACK_BEFORE_MAX", "0.5"))
SLACK_GUARD = 0.15

# Consecutive lines of the same speaker are merged only across this short a pause and up to this length.
# Longer blocks would be spoken continuously over the original's pauses, which drifts out of sync.
MERGE_MAX_GAP = float(os.environ.get("MERGE_MAX_GAP", "0.3"))
MERGE_MAX_DURATION = float(os.environ.get("MERGE_MAX_DURATION", "10.0"))
# Lines shorter than this are joined with a neighbour of the same speaker: a 0.4 s slot can only produce a
# clipped fragment of a word, so it is better spoken together with the following line.
MIN_SEGMENT_DURATION = float(os.environ.get("MIN_SEGMENT_DURATION", "1.2"))
# A take may overrun its slot by this much; the final mix speeds it up (pitch-preserving) instead of the LLM
# cutting words out of the translation.
MAX_ACCEPT_SPEEDUP = float(os.environ.get("MAX_ACCEPT_SPEEDUP", "1.08"))

# A same-speaker line that does not end a sentence is joined with the next one across this pause and up to this
# length, so the translator gets whole sentences instead of fragments (Spec 001, 2.2a).
SENTENCE_MAX_GAP = float(os.environ.get("SENTENCE_MAX_GAP", "1.5"))
SENTENCE_MAX_DURATION = float(os.environ.get("SENTENCE_MAX_DURATION", "20.0"))

# Translation (Spec 001, 2.5a): pace of the synthesized voice used for the syllable budgets, window size, and the
# completeness review pass.
DUB_SYLLABLES_PER_SECOND = float(os.environ.get("DUB_SYLLABLES_PER_SECOND", "5.0"))
TRANSLATION_WINDOW = int(os.environ.get("TRANSLATION_WINDOW", "20"))
TRANSLATION_REVIEW = os.environ.get("TRANSLATION_REVIEW", "on").lower() not in ("0", "off", "false", "no")

# Sync loop (Spec 002): a take shorter than MIN_FILL_RATIO of the original's speech is refined as too short; one
# faster than MAX_PLAUSIBLE_SYLLABLES_PER_SECOND was swallowed by the TTS and is synthesized again.
MIN_FILL_RATIO = float(os.environ.get("MIN_FILL_RATIO", "0.7"))
MAX_PLAUSIBLE_SYLLABLES_PER_SECOND = float(os.environ.get("MAX_PLAUSIBLE_SYLLABLES_PER_SECOND", "9.0"))

# Voice references (Spec 001, 2.4): level of the Demucs vocals over the rest of the mix in a clip, in dB; under this
# the crowd or the engines are as loud as the voice and OmniVoice would clone them too.
REF_MIN_CLARITY_DB = float(os.environ.get("REF_MIN_CLARITY_DB", "3.0"))


# Take check (Spec 002): a take is transcribed and compared with its text; below TAKE_MIN_SIMILARITY it is
# unintelligible (noise, screams, mangled words) and synthesized again with another voice reference.
# on = the voice audition and every take; audition = only the audition (a few minutes); off = neither.
def _take_check_mode(value: str) -> str:
    value = value.strip().lower()
    if value in ("0", "off", "false", "no"):
        return "off"
    return "audition" if value == "audition" else "on"


TAKE_CHECK = _take_check_mode(os.environ.get("TAKE_CHECK", "on"))
# cpu by default: a check model on the TTS card takes the VRAM OmniVoice needs for its peaks and makes it crash
TAKE_CHECK_DEVICE = os.environ.get("TAKE_CHECK_DEVICE", "cpu").lower()
# Every accepted take under 0.7 in the 2026-11 BattleBots run was mangled ("fuego, bo ta", squeaks before the text)
TAKE_MIN_SIMILARITY = float(os.environ.get("TAKE_MIN_SIMILARITY", "0.7"))
# CPU threads of the take-check model; faster-whisper uses 4 unless told otherwise
TAKE_CHECK_THREADS = max(1, int(os.environ.get("TAKE_CHECK_THREADS", str(os.cpu_count() or 4))))
# A line whose every take is broken below this is left undubbed: the original beats gibberish
TAKE_DROP_SIMILARITY = float(os.environ.get("TAKE_DROP_SIMILARITY", "0.3"))
# Voice audition (Spec 002, 2.3): reference candidates kept per speaker, and the score a speaker needs to use line clips
REF_AUDITION_CANDIDATES = max(1, int(os.environ.get("REF_AUDITION_CANDIDATES", "3")))
AUDITION_LINE_REF_MIN = float(os.environ.get("AUDITION_LINE_REF_MIN", "0.8"))
# One OmniVoice voice-design clip per speaker (gender/age/pitch from the profile) joins the reference candidates
VOICE_DESIGN = os.environ.get("VOICE_DESIGN", "on").lower() not in ("0", "off", "false", "no")


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
