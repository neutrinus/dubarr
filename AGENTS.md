# AGENTS.md

Guidance for AI coding agents (Claude Code, Codex, Gemini CLI, Cursor, ...) working in this repository.
Human contributors may find it useful too.

## What this project is

**dubarr** is a self-hosted service that adds AI-dubbed audio tracks ("AI - Polish", "AI - German", ...) to existing
video files. It is meant to sit next to the `*arr` stack (Radarr/Sonarr): a webhook queues a file, a background worker
runs the dubbing pipeline, and the result is muxed back into the original container in place (the original is kept as
`<file>.bak`).

Everything runs locally on the GPU(s) inside one Docker container. No cloud APIs.

## Technology stack

Python 3.12, FastAPI + SQLite + FFmpeg; AI: faster-whisper (ASR), pyannote (diarization), Demucs (separation),
OmniVoice (speech synthesis), llama.cpp / any OpenAI-compatible server (LLM). Everything runs in Docker Compose,
configured only through environment variables.

## Workflow for behaviour changes

1. Read the matching spec in `specification/specs/` (and `specification/ARCHITECTURE.md` for layering).
2. Update the spec first when a request changes behaviour, and get it accepted before writing the code.
3. Implement, with tests, following the conventions below.
4. Run the checks (below) and, for pipeline changes, the E2E script.

Small bug fixes that do not change intended behaviour (typos, crashes) need no spec change, but must not regress one.

## Repository map

```
src/
  api/server.py            FastAPI app: dashboard, /webhook, /upload, /health, /ws/logs. Starts the JobWorker in lifespan.
  server.py                Container entrypoint (uvicorn on :8000). Docker CMD runs this from /app/src.
  main.py                  AIDubber: builds the shared engine managers; also a standalone "process ./videos" mode.
  config.py                Env-var config, paths, LANG_MAP. Has import-time side effects (see Gotchas).
  prompts.py               LLM prompt templates (analysis, ASR editor, translation, duration refinement).
  utils.py                 JSON parsing, SRT cleaning, ZCR, syllable counting, output cleaning.
  omnivoice_server.py      Flask OmniVoice TTS server. Separate process on the app's interpreter, 127.0.0.1:5051.
  verify_output.py         Manual QA helper (faster-whisper re-transcription of generated segments).
  core/
    pipeline.py            DubbingPipeline: stages 1-7, step caching, Stage 5 checkpointing, final mix/mux.
    synchronizer.py        Per-segment loop: pick length variant by estimate -> TTS -> measure -> other variant / LLM -> retry.
    llm_engine.py          LLMManager (llama-cpp-python Gemma 3 12B GGUF, or an external server): analysis, ASR fix, drafts, refinement.
    tts_manager.py         TTSManager: voice reference selection (dynamic / golden / last good) + TTS calls.
    whisper_engine.py      WhisperManager (faster-whisper, large-v3-turbo).
    diarization_engine.py  DiarizationManager (pyannote speaker-diarization-community-1, needs HF_TOKEN).
    audio.py               Demucs separation, analysis orchestration, ffmpeg mixing (ducking, ambient ghost).
    gpu_services.py        LLMService / TTSService: priority-queue worker threads that serialize GPU access.
    gpu_manager.py         Picks a CUDA device by free VRAM (maps nvidia-smi -> torch index by name+memory).
    media_context.py       Parses Sonarr/Radarr/scene file names, gathers show/episode or movie info, formats it for prompts.
    worker.py              JobWorker: polls SQLite for QUEUED tasks and runs a fresh pipeline per task.
  infrastructure/
    database.py            SQLite: `tasks` queue + `job_steps` (cached JSON results per stage).
    ffmpeg.py              FFmpegWrapper: ffprobe metadata, extract/convert, mux.
    tts_client.py          OmniVoice server client: start/restart the server subprocess, HTTP calls.
    llm_client.py          OpenAI-compatible chat completions client (used when LLM_API_URL is set).
    metadata_client.py     .nfo parser, TVmaze (no key) and TMDB (optional key) clients for show/movie metadata.
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
2b. **Media metadata**: `core/media_context.py` identifies the show/episode or movie from the path (Sonarr/Radarr naming)
   and fetches synopsis and cast (character, actor, gender) from `.nfo` files, TVmaze or TMDB. Failures only log.
3. **Global analysis**: the LLM writes a summary, a phonetic glossary and speaker profiles, then corrects ASR errors in
   chunks (reference subtitles from `<video>.srt` or the first embedded subtitle stream are used as ground truth).
4. **Voice references**: picks a "golden" reference clip per speaker (scored, ZCR-checked). Despite the step name
   "Transcription Correction", this stage does not correct text.
5. **Production**: draft translations per language (windows of 20 lines with surrounding context and per-line syllable
   budgets, then a completeness review pass), then a flat pool of (segment, language) jobs runs
   through `SegmentSynchronizer` on 4 orchestrator threads. LLM and TTS calls go through `LLMService`/`TTSService`.
   Progress is checkpointed to `TEMP_DIR/stage5_checkpoint_<task_id>.json`.
6. **Mix**: per-language mastering (pan, echo, atempo), then sidechain ducking against the original track.
7. **Mux**: new AC3 tracks are added in front of the original audio; the source file is swapped in place.

Stages run through `DubbingPipeline._run_step`, which caches each result in `job_steps` and reuses it on retry.

## Setting up and running checks

Requires [uv](https://docs.astral.sh/uv/) and `ffmpeg` on PATH. The lightweight dev environment does **not** install
torch, llama-cpp, pyannote, faster-whisper or OmniVoice; those imports are optional and the code falls back to mocks.

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
real pipeline on a CPU runner using small models (Whisper `small`, Qwen2.5 3B GGUF, OmniVoice on CPU), and checks the
muxed tracks (count, tags, duration, not silent). It runs nightly, on push to `main`, on manual dispatch, and on PRs that
touch `src/`, dependencies, the workflow or the script. Without the `HF_TOKEN` secret, diarization is replaced by a single speaker. It tests
plumbing, not translation quality.

Running the real pipeline needs an NVIDIA GPU (roughly 10 GB for the LLM, about 4 GB each for Whisper and the TTS),
`HF_TOKEN` with access to the pyannote models, and the Docker image. The image has a single Python 3.12 venv
(`/app/.venv`) with everything, OmniVoice included; the TTS server is started as a subprocess on that same interpreter.
Use `docker compose up --build`. XTTS (Coqui) was removed in favour of OmniVoice; it pinned `transformers<5`, which is
why the image used to need separate venvs.

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
| `LLM_API_URL` | none | OpenAI-compatible server (`http://host:8080/v1`, e.g. llama-server). When set, no LLM is loaded in-process. |
| `LLM_API_MODEL` / `LLM_API_KEY` / `LLM_TIMEOUT` | none / none / `600` | Model name, key and per-request timeout for that server. |
| `METADATA_LOOKUP` | `on` | Show/movie context for the LLM: `on` (.nfo, then TVmaze / TMDB), `local` (.nfo only), `off`. No network in `MOCK_MODE`. |
| `TMDB_API_KEY` / `METADATA_TIMEOUT` | none / `10` | Optional TMDB key (movies; v3 key or v4 token) and per-request timeout. |
| `LLM_N_CTX` | `8192` | Context size of the in-process model. |
| `LLM_MAX_TOKENS` | `4096` | Completion budget for the bulk calls (analysis, editor, drafts); a truncated answer is retried once with double the budget. |
| `MERGE_MAX_GAP` / `MERGE_MAX_DURATION` | `0.3` / `10` | Consecutive lines of one speaker are merged only across this pause and up to this length, so the dub stays aligned with the original pauses. |
| `MIN_SEGMENT_DURATION` | `1.2` | Lines shorter than this are joined with a same-speaker neighbour; a sub-second slot can only produce a clipped fragment. |
| `MAX_ACCEPT_SPEEDUP` | `1.08` | A take may overrun its slot by this much; the mix speeds it up instead of the LLM deleting words. |
| `SENTENCE_MAX_GAP` / `SENTENCE_MAX_DURATION` | `1.5` / `20` | A same-speaker line that does not end a sentence is joined with the next across this pause, up to this length, so whole sentences get translated. |
| `SLACK_AFTER_MAX` / `SLACK_BEFORE_MAX` | `1.5` / `0.5` | How far a take may run into the silence after its line, and how much earlier a still-too-long take may start in the silence before it. |
| `DUB_SYLLABLES_PER_SECOND` | `5.0` | Pace of the synthesized voice; turns each line's time budget into a syllable budget for the translator. |
| `TRANSLATION_WINDOW` / `TRANSLATION_REVIEW` | `20` / `on` | Lines per translation request; second LLM pass that restores missing information. |
| `TTS_FILL_RATIO` / `TTS_MIN_SPEED` / `TTS_MAX_SPEEDUP` | `0.9` / `0.88` / `1.2` | OmniVoice duration control: slow short takes down towards the original's timing, speed long ones up, both only within this range. |
| `MIN_FILL_RATIO` | `0.7` | A take shorter than this share of the original's speech goes back to the LLM to restore content. |
| `MAX_PLAUSIBLE_SYLLABLES_PER_SECOND` | `9` | A faster take was swallowed by the TTS (mostly silence) and is synthesized again with another voice reference. |
| `TAKE_CHECK` / `TAKE_MIN_SIMILARITY` | `on` / `0.5` | Whisper transcribes every take; one that sounds less like its text than this (noise, screams, gibberish) is synthesized again with another voice reference. |
| `TAKE_CHECK_DEVICE` | `auto` | `auto` loads the check model on a GPU with 2.5 GB free, else the CPU; `cpu` forces the CPU. |
| `OMNIVOICE_PYTHON` | the app's interpreter | Interpreter for the OmniVoice server subprocess. |
| `TTS_DEVICE` | `auto` | `auto` puts synthesis on the GPU with the most free VRAM; `cpu` keeps the GPUs free for other work (much slower). |
| `TTS_CPU_THREADS` | all cores | Threads used when synthesis runs on the CPU. |
| `TTS_MAX_CHUNK_SECONDS` | `10` | Lines longer than this are synthesized in chunks at sentence boundaries (keeps sync and caps the VRAM peak). |
| `TTS_REF_TEXT` | `off` | Send the reference clip's transcript to OmniVoice. `off` avoids English pronunciation leaking into the target language. |
| `TTS_TIMEOUT` | `60` | Seconds per TTS request (raise it on CPU). |

## Development utilities

- `python src/main.py` (with `PYTHONPATH=.:src`) processes everything in `videos/` without the API or the DB worker —
  handy for debugging one file.
- `src/verify_output.py` re-transcribes generated segments with faster-whisper to check audio integrity by ear-free QA.
- `scripts/e2e_test.py` dubs a short clip end to end and verifies the muxed tracks (used by `.github/workflows/e2e-cpu.yml`).

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
- **Commits:** small, logically cohesive changes with `feat:` / `fix:` / `refactor:` / `docs:` / `test:` / `ux:`
  prefixes. Do not mix unrelated features, fixes and docs in one commit. Push to the working branch and confirm CI
  (lint, unit tests, smoke test, nightly E2E) went green.

## Gotchas

- `config.py` runs at import time: it prints debug paths, reads env vars, and calls `sys.exit(1)` if `HF_TOKEN` is
  missing and `MOCK_MODE` is off. Set `MOCK_MODE=1` before importing anything from `src/` in tests or scripts.
- `TEMP_DIR` is shared between tasks and its file names (`orig.wav`, `vocals.mp3`, `mono.wav`, ...) are not
  namespaced by task. Per-segment files carry the language (`raw_<lang>_<idx>.wav`, `attempt_<lang>_<idx>_<n>.wav`);
  keep it that way, because target languages run concurrently and share segment indices.
- Cached step results are JSON in SQLite. `_run_step` only re-runs a step when a cached path no longer exists, so a
  cached result can point at a file that was since overwritten by another task.
- Prompts in `prompts.py` are system prompts sent as chat messages with `response_format=json_object`; the model's chat
  template (applied by llama-cpp-python or the server) adds the special tokens. Don't put ChatML tokens in them.
- The OmniVoice server is restarted automatically on HTTP 500s and connection errors, then the request is retried once.
- `omnivoice_server.py` sets `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` before importing torch. It usually
  runs next to another GPU process (a shared `llama-server`), where the default allocator dies on 20 MB
  allocations even with memory free.
- Long lines are synthesized in ~`TTS_MAX_CHUNK_SECONDS` chunks split at sentence ends (then commas/semicolons, words
  only as a last resort). OmniVoice's VRAM peak grows with the length of the generated clip, and one long block also
  drifts out of sync with the original pauses.

## External compute node (glowonog)

Heavy work can be offloaded to the `glowonog` machine (10.252.1.122) over an OpenAI-compatible API:

| Service | Endpoint | Model | Notes |
| --- | --- | --- | --- |
| LLM (llama.cpp) | `http://10.252.1.122:8080/v1` | `gemma` | Gemma 4 26B A4B MoE, system prompts, JSON mode. No auth needed. |

Set `LLM_API_URL=http://10.252.1.122:8080/v1` and `LLM_API_MODEL=gemma` to use it instead of the in-process model.
There is no separate STT service there: transcription runs in-process with faster-whisper.
The node runs two GPUs (RTX 3060 12GB + RTX 3070 8GB) and usually hosts a long-context llama-server for the whole
house, so leave ~6GB free on the card that runs speech synthesis (see the hardware notes in `README.md`).
