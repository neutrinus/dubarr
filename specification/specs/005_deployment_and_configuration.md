# Spec 005: Deployment & Configuration

## 1. Deployment Model
Dubarr runs as one Docker container next to the `*arr` stack. Everything (ASR, diarization, separation, LLM, TTS)
runs locally; the only optional network calls are the metadata lookups (TVmaze, TMDB) and an external
OpenAI-compatible LLM server when `LLM_API_URL` is set.

### 1.1 Container
-   **Base image:** `nvidia/cuda:13.1.1-devel-ubuntu24.04`; images are published to `ghcr.io/neutrinus/dubarr` from
    `main` and tags.
-   **Single environment:** one Python 3.12 virtual environment, `/app/.venv`, with the API, faster-whisper, pyannote,
    Demucs, llama-cpp-python and OmniVoice. The OmniVoice TTS server still runs as a separate process
    (`src/omnivoice_server.py`, `127.0.0.1:5051`, started by the app on the same interpreter), so a CUDA error in the
    TTS can be recovered by restarting it without killing the worker.
-   **User:** `entrypoint.sh` creates a user for `PUID`/`PGID` and drops to it with `gosu`; the app never runs as root.
-   **Port:** the API and dashboard listen on `8000` (`PORT` overrides it).
-   **Volumes:**
    -   `/app/data` (`DATA_DIR`): `queue.db`, `logs/processing.log`, downloaded models (`models/`), per-task temp
        folders (`temp/task_<id>`) and the series memory (`series/<show>/`, Spec 006).
    -   The media library, mounted so that the paths sent to `/webhook` exist inside the container (e.g.
        `/path/to/media:/app/videos`, or the same path Sonarr/Radarr use).

### 1.2 Docker Compose
`README.md` has the recommended `docker-compose.yml` for the published image. The `docker-compose.yml` in the
repository builds the image from source for development.

## 2. Configuration
Configuration is only through environment variables (read in `src/config.py`, the duration control in
`src/duration_policy.py`). This table is the full reference; the feature specs repeat the settings that belong to them.

### 2.1 Deployment and service
| Variable | Default | Meaning |
| --- | --- | --- |
| `HF_TOKEN` | none | **Required** unless `MOCK_MODE=1`. Hugging Face token with access to the pyannote models. |
| `TARGET_LANGS` | `pl` | Comma-separated target language codes. |
| `DATA_DIR` | `/app/data` | DB, logs, models, temp files and series memory. |
| `API_USER` / `API_PASS` | `dubarr` / `dubarr` | Basic auth for `/webhook`, `/upload`, `/download`, `/debug`. |
| `PUID` / `PGID` | `1000` | UID/GID the container drops to. |
| `FIX_MEDIA_PERMISSIONS` | `0` | `1` makes the entrypoint `chown -R` the whole media mount (slow on large libraries). |
| `PORT` | `8000` | Port of the API and dashboard. |
| `DEBUG` | `0` | Keep intermediate artifacts in `videos/debug_<name>/` and the task's temp folder. |
| `VERBOSE` | `0` | DEBUG log level. |
| `MOCK_MODE` | `0` | Skip all model loading; engines return canned output. Used by tests and the smoke test. |

### 2.2 Models and devices
| Variable | Default | Meaning |
| --- | --- | --- |
| `WHISPER_MODEL` | `large-v3-turbo` | faster-whisper model name. |
| `LLM_MODEL_REPO` / `LLM_MODEL_FILE` | `bartowski/google_gemma-3-12b-it-GGUF` / `google_gemma-3-12b-it-Q4_K_M.gguf` | GGUF downloaded to `DATA_DIR/models` on first use (in-process LLM only). |
| `LLM_N_CTX` | `8192` | Context size of the in-process model. |
| `LLM_API_URL` | none | OpenAI-compatible server (`http://host:8080/v1`, e.g. llama-server). When set, no LLM is loaded in-process. |
| `LLM_API_MODEL` / `LLM_API_KEY` | none | Model name and key sent to that server (llama-server ignores both). |
| `LLM_TIMEOUT` | `600` | Seconds per request to that server. |
| `LLM_MAX_TOKENS` | `4096` | Completion budget for the bulk calls (analysis, editor, drafts); a truncated answer is retried once with double the budget. |
| `TTS_DEVICE` | `auto` | `auto`: OmniVoice on the GPU with the most free VRAM; `cpu`: keep the GPUs free (much slower). |
| `TTS_CPU_THREADS` | all cores | Threads for CPU synthesis. |
| `TTS_TIMEOUT` | `60` | Seconds per TTS request (raise it on CPU). |
| `TTS_MAX_CHUNK_SECONDS` | `10` | Longer lines are synthesized in chunks at sentence ends, then commas/semicolons (sync and VRAM peak). |
| `TTS_REF_TEXT` | `off` | Send the reference clip's transcript to OmniVoice; `off` keeps English pronunciation out of the target language. |
| `OMNIVOICE_PYTHON` | the app's interpreter | Interpreter for the OmniVoice server subprocess. |
| `AUDIO_CODEC` / `AUDIO_BITRATE` | `aac` / `192k` | Codec of the dubbed tracks: `aac` (plays in browsers, on Android and TVs), `ac3`, `eac3` or `opus`, and its bitrate. Original tracks are copied. |

`GPU_TTS` and `TTS_PORT` are set by the app for the TTS subprocess; they are not user settings.

### 2.3 Media metadata (Spec 001, 2.2b)
| Variable | Default | Meaning |
| --- | --- | --- |
| `METADATA_LOOKUP` | `on` | `on` (`.nfo`, then TVmaze for episodes / TMDB for movies), `local` (`.nfo` only), `off`. No network in `MOCK_MODE`. |
| `TMDB_API_KEY` | none | Optional TMDB key (v3 key or v4 read token) for movies. Episodes use TVmaze, which needs no account. |
| `METADATA_TIMEOUT` | `10` | Seconds per metadata request. |

### 2.4 Transcription and script (Spec 001, 2.2)
| Variable | Default | Meaning |
| --- | --- | --- |
| `ASR_SOURCE` | `original` | What Whisper transcribes: the original mix or the Demucs `vocals`. Diarization and voice references always use the vocals. |
| `ASR_GAP_PASS` / `ASR_GAP_MIN` | `on` / `2.0` | Second Whisper pass without VAD over gaps of at least this length; only confident segments are kept. |
| `MERGE_MAX_GAP` / `MERGE_MAX_DURATION` | `0.3` / `10` | Same-speaker lines are merged only across this pause and up to this length. |
| `SENTENCE_MAX_GAP` / `SENTENCE_MAX_DURATION` | `1.5` / `20` | A same-speaker line that does not end a sentence is joined with the next across this pause, up to this length. |
| `MIN_SEGMENT_DURATION` | `1.2` | Shorter lines are joined with a same-speaker neighbour. |
| `SLACK_AFTER_MAX` / `SLACK_BEFORE_MAX` | `1.5` / `0.5` | Silence after a line a take may run into, and silence before it a still-too-long take may start in. |

### 2.5 Translation (Spec 001, 2.5a)
| Variable | Default | Meaning |
| --- | --- | --- |
| `DUB_SYLLABLES_PER_SECOND` | `5.0` | Pace of the synthesized voice; turns time budgets into syllable budgets. |
| `TRANSLATION_WINDOW` | `20` | Lines per translation request. |
| `TRANSLATION_REVIEW` | `on` | Second LLM pass that restores missing information. |
| `GLOSSARY_MODE` | `respell` | `respell`: names get a phonetic spelling checked by an LLM reading test; `phonemes` (experiment): English CMU phonemes for the TTS (Spec 006, 3). |

### 2.6 Synthesis, sync loop and take judge (Spec 002)
| Variable | Default | Meaning |
| --- | --- | --- |
| `TTS_FILL_RATIO` / `TTS_MIN_SPEED` / `TTS_MAX_SPEEDUP` | `0.9` / `0.88` / `1.2` | OmniVoice duration control: slow short takes towards the original's timing, speed long ones up, within this range. |
| `MIN_FILL_RATIO` | `0.7` | A take shorter than this share of the original's speech goes back to the LLM. |
| `MAX_ACCEPT_SPEEDUP` | `1.08` | A take may overrun its limit by this much; the mix speeds it up. |
| `MAX_PLAUSIBLE_SYLLABLES_PER_SECOND` | `9` | A faster take was swallowed by the TTS (mostly silence). |
| `REF_MIN_CLARITY_DB` | `3.0` | A voice reference less than this many dB above the rest of the mix is avoided. |
| `REF_AUDITION_CANDIDATES` / `AUDITION_LINE_REF_MIN` | `3` / `0.8` | Reference clips kept per speaker for the voice audition; a speaker scoring below this does not clone from line clips. |
| `VOICE_DESIGN` | `on` | One OmniVoice voice-design clip per speaker joins the reference candidates. |
| `TAKE_CHECK` | `judge` | `judge`: the take judge decides on the audition and every take; `whisper`: the same plus the Whisper similarity in the report, for calibration; `off`: only the swallowed and signal checks, no audition. The old values `on` and `audition` read as `judge`. |
| `TAKE_VERSIONS` | `2` | Takes per synthesis, from different references; the best is kept. |
| `JUDGE_MIN_TEXT` / `JUDGE_MIN_VOICE` / `JUDGE_HF_MAX_DB` | `0.5` / `0.5` / `12` | Judge thresholds (first guesses, to be tuned with `TAKE_CHECK=whisper`). |
| `JUDGE_ALIGN_MODEL` / `JUDGE_DEVICE` | per language / `cpu` | CTC model of the text match and where the judge runs. |
| `TAKE_DROP_SIMILARITY` | `0.3` | A line whose every take is broken below this score is left undubbed. |
| `TAKE_MIN_SIMILARITY` / `TAKE_CHECK_DEVICE` / `TAKE_CHECK_THREADS` | `0.7` / `cpu` / all cores | Whisper similarity of `TAKE_CHECK=whisper`, its device (`auto` uses a GPU only with 8.5 GB free) and CPU threads. |

### 2.7 Series memory (Spec 006)
| Variable | Default | Meaning |
| --- | --- | --- |
| `SERIES_MEMORY` | `on` | Episodes of a series share settled name spellings and a voice bank in `DATA_DIR/series/<show>/`. |
| `VOICE_MATCH_MIN` / `VOICE_BANK_CLIPS` | `0.6` / `5` | Voiceprint similarity from which a speaker is a known person; clips kept per person. |

## 3. GPU Resource Management
-   **Lazy loading:** Demucs, pyannote and Whisper are loaded for their stage and unloaded right after it, so the LLM
    and the TTS get the VRAM in Stage 3 and Stage 5.
-   **Device mapping:** `GPUManager.get_best_gpu` maps `nvidia-smi` indices to torch `cuda:X` indices by name and
    memory size, and picks the card with the most free VRAM.
-   **Single GPU or CPU:** an inference lock serializes the LLM, Whisper and TTS so they never run at the same time.
-   **LLM:** all calls are chat completions with JSON output, so the model's own chat template applies. With
    `LLM_API_URL` the LLM runs in an external server and dubarr loads none.
-   **Stage 5 services:** LLM and TTS calls go through `LLMService` / `TTSService`, worker threads with priority queues.
    `LLMService` has exactly 1 worker, because concurrent CUDA inference in llama-cpp-python segfaults.

## 4. Hardware Requirements
-   **GPU:** strongly recommended (NVIDIA, CUDA). About 10 GB for the in-process LLM (Gemma 3 12B, 8k context),
    about 4 GB for Whisper and diarization (unloaded after Stage 2), and up to 6 GB for OmniVoice on long lines.
-   **Two cards** (e.g. 12 GB + 8 GB): the LLM on one, the TTS on the other. When the cards are shared with another
    service, keep about 6 GB free on the card that runs the TTS, or set `TTS_DEVICE=cpu`.
-   **RAM:** 16 GB minimum, 32 GB for CPU-only runs.

## 5. Local Development
The dev environment (`uv sync --group dev`) has no torch, llama-cpp, pyannote, faster-whisper or OmniVoice; the
engines fall back to mocks with `MOCK_MODE=1`. Running the real pipeline outside Docker needs one Python 3.12
environment: `uv sync --group runtime` (plus `llama-cpp-python` when the LLM runs in-process). The app starts the
OmniVoice server itself.
