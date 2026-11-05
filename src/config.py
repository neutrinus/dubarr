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
# CPU threads of the in-process model. More threads than cores make llama.cpp spin: 8 threads on a 4-core
# runner ran at ~1.7 tokens/s.
LLM_THREADS = int(os.environ.get("LLM_THREADS", "0")) or min(8, os.cpu_count() or 1)
# Per-request completion budget for the bulk calls (analysis, ASR editor, drafts). A truncated answer
# triggers one retry with double the budget, so a reasoning model cannot silently swallow the whole reply.
LLM_MAX_TOKENS = int(os.environ.get("LLM_MAX_TOKENS", "4096"))

print(f"DEBUG: config.BASE_DIR={BASE_DIR}", flush=True)
print(f"DEBUG: config.DATA_DIR={DATA_DIR}", flush=True)
print(f"DEBUG: config.VIDEO_FOLDER={VIDEO_FOLDER}", flush=True)
print(f"DEBUG: config.OUTPUT_FOLDER={OUTPUT_FOLDER}", flush=True)
print(f"DEBUG: config.MODEL_PATH={MODEL_PATH}", flush=True)

# Full large-v3: on BattleBots S01E01 it heard 14% more segments than turbo, spelled the names right ("Ice Wave",
# not "Icewing") and wrote no "We'll be right back." over music; it costs ~3 minutes per episode (Spec 001, 2.2).
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "large-v3")
# The take check of TAKE_CHECK=whisper runs on the CPU, where turbo is several times faster
TAKE_CHECK_MODEL = os.environ.get("TAKE_CHECK_MODEL", "large-v3-turbo")

# Demucs gets a whole episode in one call; a 45-minute 720p episode takes ~8 minutes on a 3060, so the limit
# must leave room for slower cards and larger files.
DEMUCS_TIMEOUT = int(os.environ.get("DEMUCS_TIMEOUT", "1800"))
# Second transcription pass without VAD over the gaps the first pass left (commentary under arena noise, which VAD
# drops); only confident, non-hallucinated segments are kept (Spec 001, 2.2).
# What Whisper transcribes: the original mix (default) or the Demucs vocals; diarization always uses the vocals
ASR_SOURCE = os.environ.get("ASR_SOURCE", "original").lower()
ASR_GAP_PASS = os.environ.get("ASR_GAP_PASS", "on").lower() not in ("0", "off", "false", "no")
ASR_GAP_MIN = float(os.environ.get("ASR_GAP_MIN", "2.0"))
# Script verification (Spec 001, 2.2c): unsure segments are heard again, the corrector picks or reconstructs the line
ASR_VERIFY = os.environ.get("ASR_VERIFY", "on").lower() not in ("0", "off", "false", "no")
ASR_UNSURE_LOGPROB = float(os.environ.get("ASR_UNSURE_LOGPROB", "-0.6"))
ASR_UNSURE_WORD = float(os.environ.get("ASR_UNSURE_WORD", "0.3"))
# A segment closer to another speaker's voiceprint by this margin (and at least this similar) is that speaker's
SPEAKER_RECHECK_MARGIN = float(os.environ.get("SPEAKER_RECHECK_MARGIN", "0.1"))
SPEAKER_RECHECK_MIN = float(os.environ.get("SPEAKER_RECHECK_MIN", "0.5"))

# Seconds per TTS request (raise it on CPU, synthesis is slow there)
TTS_TIMEOUT = int(os.environ.get("TTS_TIMEOUT", "60"))
# How long a TTS request waits for GPU memory when the card is full (another process on it), before it fails
TTS_OOM_WAIT = int(os.environ.get("TTS_OOM_WAIT", "300"))

# Speech synthesis is OmniVoice. Its server runs as a separate process, on the app's own interpreter by default.
if os.environ.get("TTS_BACKEND", "omnivoice").lower() != "omnivoice":
    print("WARNING: TTS_BACKEND is no longer used, XTTS was removed; OmniVoice is the only TTS backend.", flush=True)
OMNIVOICE_PYTHON = os.environ.get("OMNIVOICE_PYTHON", sys.executable)

# Codec and bitrate of the dubbed tracks: aac (default, plays in browsers and on phones), ac3, eac3 or opus.
# The original audio tracks are copied unchanged.
AUDIO_CODEC = os.environ.get("AUDIO_CODEC", "aac").lower()
AUDIO_BITRATE = os.environ.get("AUDIO_BITRATE", "192k")

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


# Take check (Spec 002, 2.2 step 4): judge = the take judge (signal artifacts, voice match, CTC text match) on the
# voice audition and every take; whisper = the judge plus the Whisper similarity next to it, for measurement only;
# off = only the swallowed and artifact checks. The judge is cheap, so the old "on" and "audition" read as judge.
def _take_check_mode(value: str) -> str:
    value = value.strip().lower()
    if value in ("0", "off", "false", "no"):
        return "off"
    return "whisper" if value == "whisper" else "judge"


TAKE_CHECK = _take_check_mode(os.environ.get("TAKE_CHECK", "judge"))
# Takes per synthesis, from different voice references; the judge keeps the best
TAKE_VERSIONS = max(1, int(os.environ.get("TAKE_VERSIONS", "4")))
# Short lines (4 to SHORT_LINE_SYLLABLES syllables: shouts, names) get more takes; the TTS slurs them most often
TAKE_VERSIONS_SHORT = max(1, int(os.environ.get("TAKE_VERSIONS_SHORT", "6")))
SHORT_LINE_SYLLABLES = int(os.environ.get("SHORT_LINE_SYLLABLES", "10"))
# A sound take scoring at least this ends the search for a line early
TAKE_GOOD_SCORE = float(os.environ.get("TAKE_GOOD_SCORE", "0.95"))
# Judge layers. Thresholds are first guesses, to be tuned on runs with TAKE_CHECK=whisper
JUDGE_HF_MAX_DB = float(os.environ.get("JUDGE_HF_MAX_DB", "12"))
JUDGE_MIN_VOICE = float(os.environ.get("JUDGE_MIN_VOICE", "0.35"))
JUDGE_MIN_TEXT = float(os.environ.get("JUDGE_MIN_TEXT", "0.5"))
JUDGE_ALIGN_MODEL = os.environ.get("JUDGE_ALIGN_MODEL", "")  # empty: per language (core/take_judge.py)
JUDGE_DEVICE = os.environ.get("JUDGE_DEVICE", "cpu").lower()
# Voice gender classifier (Spec 002, 2.3), the last resort when neither subtitles, cast nor a named profile tell; "off"
GENDER_MODEL = os.environ.get("GENDER_MODEL", "audeering/wav2vec2-large-robust-24-ft-age-gender")
GENDER_MIN_CONFIDENCE = float(os.environ.get("GENDER_MIN_CONFIDENCE", "0.9"))
# So sure that the voice overrules the cast list and the analysis (a name the analysis gave may be someone else's)
GENDER_VOICE_CONFIDENCE = float(os.environ.get("GENDER_VOICE_CONFIDENCE", "0.95"))
# cpu by default: a check model on the TTS card takes the VRAM OmniVoice needs for its peaks and makes it crash
TAKE_CHECK_DEVICE = os.environ.get("TAKE_CHECK_DEVICE", "cpu").lower()
# Every accepted take under 0.7 in the 2026-11 BattleBots run was mangled ("fuego, bo ta", squeaks before the text)
TAKE_MIN_SIMILARITY = float(os.environ.get("TAKE_MIN_SIMILARITY", "0.7"))
# CPU threads of the take-check model; faster-whisper uses 4 unless told otherwise
TAKE_CHECK_THREADS = max(1, int(os.environ.get("TAKE_CHECK_THREADS", str(os.cpu_count() or 4))))
# A line whose every take is broken below this is left undubbed: the original beats gibberish
TAKE_DROP_SIMILARITY = float(os.environ.get("TAKE_DROP_SIMILARITY", "0.5"))
# Voice audition (Spec 002, 2.3): reference candidates kept per speaker, and the score a speaker needs to use line clips
REF_AUDITION_CANDIDATES = max(1, int(os.environ.get("REF_AUDITION_CANDIDATES", "3")))
AUDITION_LINE_REF_MIN = float(os.environ.get("AUDITION_LINE_REF_MIN", "0.8"))
# One OmniVoice voice-design clip per speaker (gender/age/pitch from the profile) joins the reference candidates
VOICE_DESIGN = os.environ.get("VOICE_DESIGN", "on").lower() not in ("0", "off", "false", "no")

# Series memory (Spec 006): settled name spellings and the voice bank, kept between the episodes of a series
SERIES_MEMORY = os.environ.get("SERIES_MEMORY", "on").lower() not in ("0", "off", "false", "no")
VOICE_MATCH_MIN = float(os.environ.get("VOICE_MATCH_MIN", "0.6"))
VOICE_BANK_CLIPS = max(1, int(os.environ.get("VOICE_BANK_CLIPS", "5")))
VOICE_BANK_CLASS_CLIPS = max(1, int(os.environ.get("VOICE_BANK_CLASS_CLIPS", "2")))
# Reference clips by energy class (calm / lively / shout) and the merging of a shouting cluster with the same person's
# calm one (Spec 001, 2.4; Spec 002, 2.3)
EMOTION_REFS = os.environ.get("EMOTION_REFS", "on").lower() not in ("0", "off", "false", "no")
SHOUT_MERGE_MIN = float(os.environ.get("SHOUT_MERGE_MIN", "0.3"))
# How names keep their sound: respell = a phonetic spelling in the target language (checked by a reading test);
# phonemes = the translation keeps the original name and the TTS gets its English CMU phonemes ("[R EY1 Z ER0 ...]").
# phonemes is an experiment: OmniVoice documents them for English text only.
GLOSSARY_MODE = "phonemes" if os.environ.get("GLOSSARY_MODE", "respell").strip().lower() == "phonemes" else "respell"


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

# ISO 639-2 codes: the language tags of the dubbed tracks (MP4 drops two-letter codes) and of existing tracks
ISO_639_2 = {
    "pl": "pol",
    "en": "eng",
    "de": "ger",
    "es": "spa",
    "fr": "fra",
    "it": "ita",
    "ru": "rus",
    "ja": "jpn",
    "zh": "chi",
    "ko": "kor",
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
