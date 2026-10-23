# AGENTS.md

Guidance for AI coding agents (Claude Code, Codex, Gemini CLI, Cursor, ...) working in this repository.
Human contributors may find it useful too. `GEMINI.md` holds older, Gemini-specific notes; where the two differ, this file wins.

## What this project is

**dubarr** is a self-hosted service that adds AI-dubbed audio tracks ("AI - Polish", "AI - German", ...) to existing
video files. It is meant to sit next to the `*arr` stack (Radarr/Sonarr): a webhook queues a file, a background worker
runs the dubbing pipeline, and the result is muxed back into the original container in place (the original is kept as
`<file>.bak`).

Everything runs locally on the GPU(s) inside one Docker container. No cloud APIs.

## Repository map

```
src/
  api/server.py            FastAPI app: dashboard, /webhook, /upload, /health, /ws/logs. Starts the JobWorker in lifespan.
  server.py                Container entrypoint (uvicorn on :8000). Docker CMD runs this from /app/src.
  main.py                  AIDubber: builds the shared engine managers; also a standalone "process ./videos" mode.
  config.py                Env-var config, paths, LANG_MAP. Has import-time side effects (see Gotchas).
  prompts.py               LLM prompt templates (analysis, ASR editor, translation, duration refinement).
  utils.py                 JSON parsing, SRT cleaning, ZCR, syllable counting, output cleaning.
  tts_server.py            Flask XTTS v2 server. Runs in a SEPARATE Python 3.10 venv on 127.0.0.1:5050.
  omnivoice_server.py      Flask OmniVoice server (default TTS backend). Own Python 3.12 venv, 127.0.0.1:5051.
  verify_output.py         Manual QA helper (faster-whisper re-transcription of generated segments).
  core/
    pipeline.py            DubbingPipeline: stages 1-7, step caching, Stage 5 checkpointing, final mix/mux.
    synchronizer.py        Per-segment loop: TTS -> measure duration -> ask LLM to shorten -> retry (max 3).
    llm_engine.py          LLMManager (llama-cpp-python, Gemma 3 12B GGUF): analysis, ASR correction, drafts, refinement.
    tts_manager.py         TTSManager: voice reference selection (dynamic / golden / last good) + TTS calls.
    whisper_engine.py      WhisperManager (faster-whisper, large-v3-turbo).
    diarization_engine.py  DiarizationManager (pyannote speaker-diarization-community-1, needs HF_TOKEN).
    audio.py               Demucs separation, analysis orchestration, ffmpeg mixing (ducking, ambient ghost).
    gpu_services.py        LLMService / TTSService: priority-queue worker threads that serialize GPU access.
    gpu_manager.py         Picks a CUDA device by free VRAM (maps nvidia-smi -> torch index by name+memory).
    worker.py              JobWorker: polls SQLite for QUEUED tasks and runs a fresh pipeline per task.
  infrastructure/
    database.py            SQLite: `tasks` queue + `job_steps` (cached JSON results per stage).
    ffmpeg.py              FFmpegWrapper: ffprobe metadata, extract/convert, mux.
    tts_client.py          TTS server clients (XTTS, OmniVoice): start/restart the server subprocess, HTTP calls.
    monitor.py             Periodic CPU/RAM/GPU/queue logging thread.
  templates/index.html     Dashboard.
tests/                     pytest unit tests (everything heavy is mocked).
scripts/                   smoke_test.py (used by CI against a MOCK_MODE container), demo/LLM helpers.
specification/             ARCHITECTURE.md + numbered feature specs (the intended behaviour).
```

## Pipeline at a glance

1. **Separation**: ffmpeg extracts stereo, Demucs `htdemucs_ft` isolates vocals.
2. **Analysis**: pyannote diarization, then faster-whisper transcription (models are loaded and unloaded one at a time).
   `DubbingPipeline._create_script` assigns speakers to Whisper segments and merges short same-speaker lines.
3. **Global analysis**: the LLM writes a summary, a phonetic glossary and speaker profiles, then corrects ASR errors in
   chunks (reference subtitles from `<video>.srt` or the first embedded subtitle stream are used as ground truth).
4. **Voice references**: picks a "golden" reference clip per speaker (scored, ZCR-checked). Despite the step name
   "Transcription Correction", this stage does not correct text.
5. **Production**: draft translations per language (batches of 10), then a flat pool of (segment, language) jobs runs
   through `SegmentSynchronizer` on 4 orchestrator threads. LLM and TTS calls go through `LLMService`/`TTSService`.
   Progress is checkpointed to `TEMP_DIR/stage5_checkpoint_<task_id>.json`.
6. **Mix**: per-language mastering (pan, echo, atempo), then sidechain ducking against the original track.
7. **Mux**: new AC3 tracks are added in front of the original audio; the source file is swapped in place.

Stages run through `DubbingPipeline._run_step`, which caches each result in `job_steps` and reuses it on retry.

## Setting up and running checks

Requires [uv](https://docs.astral.sh/uv/) and `ffmpeg` on PATH. The lightweight dev environment does **not** install
torch, llama-cpp, pyannote, faster-whisper or Coqui; those imports are optional and the code falls back to mocks.

```bash
uv sync --group dev --python 3.10

# Unit tests: MOCK_MODE=1 is required (otherwise config.py exits without HF_TOKEN).
# Point DATA_DIR somewhere writable; the default is /app/data.
MOCK_MODE=1 DATA_DIR=/tmp/dubarr-data PYTHONPATH=.:src uv run pytest tests/

# Lint (all three run in CI; ruff and flake8 must pass, pylint is informational)
uv run ruff check .
uv run ruff format --check .
uv run flake8 .
```

CI (`.github/workflows/pipeline.yml`) runs the same lint + tests on Python 3.10, then builds the Docker image and runs
`scripts/smoke_test.py` against a container started with `MOCK_MODE=1`. Images are pushed to `ghcr.io` only from `main`
and tags.

`.github/workflows/e2e-cpu.yml` runs `scripts/e2e_test.py`: it cuts a 60 s public-domain clip (NASA), dubs it with the
real pipeline on a CPU runner using small models (Whisper `small`, Qwen2.5 1.5B GGUF, XTTS v2 on CPU), and checks the
muxed tracks (count, tags, duration, not silent). It runs nightly, on push to `main`, on manual dispatch, and on PRs that
touch `src/`, dependencies, the workflow or the script. Without the `HF_TOKEN` secret, diarization is replaced by a single speaker. It tests
plumbing, not translation quality.

Running the real pipeline needs an NVIDIA GPU (roughly 10 GB for the LLM, about 4 GB each for Whisper and XTTS),
`HF_TOKEN` with access to the pyannote models, and the Docker image (two venvs: `.venv_app` on Python 3.12 and
`.venv_tts` on Python 3.10 for Coqui). Use `docker compose up --build`. Outside Docker you must start
`src/tts_server.py` yourself from a Python 3.10 venv with Coqui installed.

## Configuration (environment variables)

| Variable | Default | Meaning |
| --- | --- | --- |
| `TARGET_LANGS` | `pl` | Comma-separated target language codes. |
| `HF_TOKEN` | none | Required unless `MOCK_MODE=1` (pyannote). |
| `DATA_DIR` | `/app/data` | DB (`queue.db`), logs, models, temp files. |
| `MOCK_MODE` | `0` | Skip all model loading; engines return canned output. Used by tests and the smoke test. |
| `DEBUG` | `0` | Keep intermediate artifacts in `videos/debug_<name>/`. |
| `VERBOSE` | `0` | DEBUG log level. |
| `API_USER` / `API_PASS` | `dubarr` / `dubarr` | Basic auth for `/webhook`, `/upload`, `/download`, `/debug`. |
| `PUID` / `PGID` | `1000` | Container user, handled by `entrypoint.sh`. |
| `WHISPER_MODEL` | `large-v3-turbo` | faster-whisper model name. |
| `LLM_MODEL_REPO` / `LLM_MODEL_FILE` | Gemma 3 12B Q4_K_M (bartowski) | GGUF downloaded to `DATA_DIR/models` on first use. |
| `TTS_BACKEND` | `omnivoice` | Speech synthesis backend: `omnivoice` or `xtts`. |
| `TTS_PYTHON` / `OMNIVOICE_PYTHON` | `/app/.venv_tts/...`, `/app/.venv_omnivoice/...` | Interpreters for the TTS servers. |
| `TTS_TIMEOUT` | `60` | Seconds per XTTS request (raise it on CPU). |

## Conventions

- **Spec first for behaviour changes.** `specification/specs/*.md` describes intended behaviour. When a change alters
  behaviour, update the matching spec in the same PR. Pure bug fixes don't need a spec change.
- **Layering** (see `specification/ARCHITECTURE.md`): HTTP and auth only in `src/api`; pipeline logic in `src/core`;
  subprocesses, DB and HTTP clients in `src/infrastructure`. In practice `core/` still calls `subprocess`/ffmpeg directly
  in several places; prefer adding to `FFmpegWrapper` over adding more of that.
- **Imports are flat from `src/`** (`from config import ...`, `from core.pipeline import ...`). Code runs with
  `PYTHONPATH=.:src` and the Docker workdir is `/app/src`. Don't switch to `src.`-prefixed imports piecemeal.
- **Every heavy engine must honour `MOCK_MODE`** and must import its ML library inside `try/except ImportError`, so the
  dev environment and CI stay light.
- **Long loops must check `abort_event`.** Every subprocess call should have a `timeout`.
- **GPU work goes through the services.** In Stage 5, LLM and TTS calls must be submitted via `LLMService`/`TTSService`.
  `LLMService` must stay at 1 worker: llama-cpp-python segfaults with concurrent CUDA inference.
- **GPU selection**: never use `nvidia-smi` indices as torch device indices; use `GPUManager.get_best_gpu`.
- Style: ruff, line length 127. Commit messages use `feat:`, `fix:`, `refactor:`, `docs:`, `test:`, `ux:` prefixes,
  in small, focused commits.
- Tests live in `tests/test_<module>.py`, use `unittest.mock` heavily, and must not need a GPU, network or model files.
  Add or update tests together with logic changes.

## Gotchas

- `config.py` runs at import time: it prints debug paths, reads env vars, and calls `sys.exit(1)` if `HF_TOKEN` is
  missing and `MOCK_MODE` is off. Set `MOCK_MODE=1` before importing anything from `src/` in tests or scripts.
- `TEMP_DIR` is shared between tasks and file names inside it (`orig.wav`, `vocals.mp3`, `raw_<idx>.wav`,
  `attempt_<idx>_<n>.wav`, ...) are not namespaced by task or language. Keep this in mind when touching caching,
  resume or multi-language code.
- Cached step results are JSON in SQLite. `_run_step` only re-runs a step when a cached path no longer exists, so a
  cached result can point at a file that was since overwritten by another task.
- Prompts in `prompts.py` are raw completion prompts (no chat template is applied by llama-cpp here).
- The XTTS server is restarted automatically on most HTTP 500s; "size of tensor ... must match" errors are treated as
  input errors and retried with a different reference instead.
- The Dockerfile patches Coqui's `autoregressive.py` with `sed` for a transformers incompatibility. If you bump Coqui
  or transformers, check that patch still applies.
